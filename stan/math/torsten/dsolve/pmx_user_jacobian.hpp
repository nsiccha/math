#ifndef STAN_MATH_TORSTEN_DSOLVE_PMX_USER_JACOBIAN_HPP
#define STAN_MATH_TORSTEN_DSOLVE_PMX_USER_JACOBIAN_HPP

#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/functor/ode_jacobian_pair.hpp>
#include <stan/math/rev/functor/ode_user_jacobian.hpp>
#include <ostream>
#include <type_traits>
#include <utility>
#include <vector>

namespace torsten {

  /**
   * Whether a right hand side functor provides <code>jacobian(...)</code>,
   * the Jacobian of its value with respect to its state and arguments in
   * Stan Math's user-Jacobian layout (see
   * <code>stan::math::internal::ode_jacobian_pair</code>). A functor
   * declares it with <code>static constexpr bool has_user_jacobian</code>.
   */
  template <typename F, typename = void>
  struct pmx_has_user_jacobian : std::false_type {};

  template <typename F>
  struct pmx_has_user_jacobian<F, std::enable_if_t<std::decay_t<F>::has_user_jacobian>>
    : std::true_type {};

  /**
   * The user's right hand side f of a PMX ODE or DDE together with its
   * Jacobian jac, which takes the same arguments as f.
   *
   * Calling the pair calls f, so the pair stands in for f in Torsten's
   * events and models; the sensitivity systems
   * (<code>dsolve::PMXVariadicOdeSystem</code>, <code>dsolve::PMXDdeSystem</code>)
   * detect it and evaluate jac in doubles in place of nested autodiff.
   *
   * <ul>
   * <li>ODE, f(t, y, theta, x_r, x_i, msgs) (or with msgs third): jac
   *   returns the N x (N + P) matrix [df/dy | df/dargs].</li>
   * <li>DDE, f(t, y, y_lag, theta, x_r, x_i, msgs): jac returns the
   *   N x (N + N D + P) matrix [df/dy | df/dy_lag per delay | df/dargs].</li>
   * </ul>
   * P counts the real scalars of the arguments in <code>count_vars</code>
   * order: theta, then x_r. The x_r columns are never read; x_i has none.
   *
   * The pair keeps references and lives for the duration of one solver
   * call.
   */
  template <typename F, typename FJac>
  struct PMXJacobianPair {
    static constexpr bool has_user_jacobian = true;
    const F& f_;
    const FJac& jac_;

    PMXJacobianPair(const F& f, const FJac& jac) : f_(f), jac_(jac) {}

    template <typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(std::declval<const F&>()(std::forward<Args>(args)...)) {
      return f_(std::forward<Args>(args)...);
    }
  };

  namespace dsolve {

    /**
     * Calls <code>f.jacobian(args...)</code>: the Jacobian functor of
     * <code>stan::math::internal::ode_jacobian_pair</code> for a right hand
     * side that provides its own Jacobian.
     */
    template <typename F>
    struct PMXJacobianMember {
      const F& f_;
      explicit PMXJacobianMember(const F& f) : f_(f) {}

      template <typename... Args>
      Eigen::MatrixXd operator()(Args&&... args) const {
        return f_.jacobian(std::forward<Args>(args)...);
      }
    };

    /**
     * The Jacobian of a right hand side f(t, y, msgs, args...) that provides
     * its own (see <code>pmx_has_user_jacobian</code>), at doubles, through
     * <code>stan::math::internal::eval_user_jacobian</code>: size checks and,
     * with <code>STAN_MATH_CHECK_USER_JACOBIAN</code>, a comparison with
     * nested autodiff.
     *
     * @tparam Origs original types of the arguments
     * @param function name for error messages
     * @param f right hand side
     * @param t time
     * @param states the states, as doubles (y; or y, y_lag)
     * @param msgs message stream
     * @param args the arguments, as doubles
     * @param num_arg_columns number of real scalars of the arguments
     * @param var_columns argument columns of the autodiff scalars
     */
    template <typename... Origs, typename F, typename... States, typename... Args>
    inline Eigen::MatrixXd
    pmx_user_jacobian(const std::tuple<Origs...>* origs, const char* function,
                      const F& f, double t, const std::tuple<States...>& states,
                      std::ostream* msgs, const std::tuple<Args...>& args,
                      size_t num_arg_columns,
                      const std::vector<size_t>& var_columns) {
      const stan::math::internal::ode_jacobian_pair<F, PMXJacobianMember<F>>
          f_jac(f, PMXJacobianMember<F>(f));
      return stan::math::internal::eval_user_jacobian(
          origs, function, f_jac, t, states, msgs, args, num_arg_columns,
          var_columns);
    }

    /**
     * The user's ODE Jacobian, called with f's argument order: the stream
     * last, f(t, y, theta, x_r, x_i, msgs), or third,
     * f(t, y, msgs, theta, x_r, x_i), whichever jac accepts.
     */
    template <typename FJac>
    inline Eigen::MatrixXd
    pmx_call_ode_user_jacobian(const FJac& jac, double t, const Eigen::VectorXd& y,
                               std::ostream* msgs, const std::vector<double>& theta,
                               const std::vector<double>& x_r,
                               const std::vector<int>& x_i) {
      if constexpr (std::is_invocable<const FJac&, double, const Eigen::VectorXd&,
                                      const std::vector<double>&,
                                      const std::vector<double>&,
                                      const std::vector<int>&,
                                      std::ostream*>::value) {
        return jac(t, y, theta, x_r, x_i, msgs);
      } else {
        return jac(t, y, msgs, theta, x_r, x_i);
      }
    }

    /**
     * Checks the size of a user Jacobian before Torsten reads it.
     */
    inline void pmx_check_user_jacobian_size(const char* function,
                                             const Eigen::MatrixXd& jac,
                                             size_t rows, size_t cols) {
      stan::math::check_size_match(function, "rows of the Jacobian", jac.rows(),
                                   "states", rows);
      stan::math::check_size_match(function, "columns of the Jacobian", jac.cols(),
                                   "state and argument scalars", cols);
    }

    /**
     * A DDE right hand side or Jacobian g(t, y, y_lag, args..., msgs),
     * called in Stan Math's order g(t, y, y_lag, msgs, args...).
     */
    template <typename G>
    struct PMXDdeStreamLast {
      const G& g_;
      explicit PMXDdeStreamLast(const G& g) : g_(g) {}

      template <typename T0, typename T1, typename T2, typename... Args>
      auto operator()(const T0& t, const T1& y, const T2& y_lag,
                      std::ostream* msgs, const Args&... args) const {
        return g_(t, y, y_lag, args..., msgs);
      }
    };

  }  // namespace dsolve
}  // namespace torsten

#endif
