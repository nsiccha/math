#ifndef STAN_MATH_PRIM_FUNCTOR_COUPLED_DDE_SYSTEM_HPP
#define STAN_MATH_PRIM_FUNCTOR_COUPLED_DDE_SYSTEM_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/prim/functor/apply.hpp>
#include <ostream>
#include <tuple>

namespace stan {
namespace math {

/**
 * The right hand side of a constant-delay DDE together with its forward
 * sensitivities, as integrated by <code>internal::dde_retard_integrator</code>.
 *
 * The primary template is specialized for arithmetic arguments (below, no
 * sensitivities) and for autodiff arguments (in rev).
 *
 * @tparam ArithmeticArguments whether all of y0, tau and args are arithmetic
 * @tparam F base DDE right hand side functor
 * @tparam T_y0 scalar type of the initial state
 * @tparam T_tau type of the delay vector
 * @tparam Args types of the extra arguments of F
 */
template <bool ArithmeticArguments, typename F, typename T_y0, typename T_tau,
          typename... Args>
struct coupled_dde_system_impl;

/**
 * The coupled DDE system for arithmetic arguments: there are no
 * sensitivities, so the coupled system is the base system.
 *
 * @tparam F base DDE right hand side functor. Must provide
 *   <code>operator()(double t, const Eigen::VectorXd& y,
 *     const Eigen::MatrixXd& y_lag, std::ostream* msgs,
 *     const Args&... args)</code>
 *   returning the derivative vector, where column j of y_lag is
 *   y(t - tau_j).
 */
template <typename F, typename T_y0, typename T_tau, typename... Args>
struct coupled_dde_system_impl<true, F, T_y0, T_tau, Args...> {
  const F& f_;
  const Eigen::VectorXd& y0_;
  std::tuple<const Args&...> args_tuple_;
  const size_t N_;
  const size_t D_;
  std::ostream* msgs_;

  /**
   * @param[in] f the base DDE system functor
   * @param[in] y0 the initial state of the base DDE
   * @param[in] tau the delays
   * @param[in, out] msgs stream for messages
   * @param[in] args other additional arguments
   */
  coupled_dde_system_impl(const F& f, const Eigen::VectorXd& y0,
                          const T_tau& tau, std::ostream* msgs,
                          const Args&... args)
      : f_(f),
        y0_(y0),
        args_tuple_(args...),
        N_(y0.size()),
        D_(tau.size()),
        msgs_(msgs) {}

  /**
   * Evaluate the right hand side of the base system.
   *
   * @param[in] t time
   * @param[in] z state
   * @param[in] z_lag lagged states; column j is z(t - tau_j)
   * @param[in] z_lag_deriv unused (no lagged derivatives are needed)
   * @param[out] dz_dt derivative of the state
   * @throw exception if the base DDE function does not return the
   *    expected number of derivatives, N.
   */
  void operator()(double t, const Eigen::VectorXd& z,
                  const Eigen::MatrixXd& z_lag,
                  const Eigen::MatrixXd& z_lag_deriv,
                  Eigen::VectorXd& dz_dt) const {
    Eigen::VectorXd f_y_t = math::apply(
        [&](const Args&... args) { return f_(t, z, z_lag, msgs_, args...); },
        args_tuple_);

    check_size_match("coupled_dde_system", "dy_dt", f_y_t.size(), "states", N_);

    dz_dt = f_y_t;
  }

  /**
   * Returns the size of the coupled system.
   */
  size_t size() const { return N_; }

  /**
   * Returns the number of states of the base system.
   */
  size_t num_states() const { return N_; }

  /**
   * Returns the derivatives of the delays with respect to the sensitivity
   * parameters: there are none.
   */
  Eigen::MatrixXd delay_sensitivities() const { return Eigen::MatrixXd(0, D_); }

  /**
   * Returns the number of leading components whose lagged derivatives the
   * right hand side needs: none.
   */
  size_t num_lag_derivatives() const { return 0; }

  /**
   * Returns the initial state of the coupled system, which is also its
   * constant initial history.
   */
  Eigen::VectorXd initial_state() const { return y0_; }
};

/**
 * The coupled DDE system: the base system with forward sensitivities with
 * respect to every autodiff variable in the initial state, the delays and
 * the extra arguments, if there are any.
 *
 * @tparam F base DDE right hand side functor
 * @tparam T_y0 scalar type of the initial state
 * @tparam T_tau type of the delay vector
 * @tparam Args types of the extra arguments of F
 */
template <typename F, typename T_y0, typename T_tau, typename... Args>
struct coupled_dde_system
    : public coupled_dde_system_impl<
          std::is_arithmetic<return_type_t<T_y0, T_tau, Args...>>::value, F,
          T_y0, T_tau, Args...> {
  coupled_dde_system(const F& f,
                     const Eigen::Matrix<T_y0, Eigen::Dynamic, 1>& y0,
                     const T_tau& tau, std::ostream* msgs, const Args&... args)
      : coupled_dde_system_impl<
          std::is_arithmetic<return_type_t<T_y0, T_tau, Args...>>::value, F,
          T_y0, T_tau, Args...>(f, y0, tau, msgs, args...) {}
};

}  // namespace math
}  // namespace stan
#endif
