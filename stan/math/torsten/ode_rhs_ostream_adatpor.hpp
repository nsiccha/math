#ifndef STAN_MATH_TORSTEN_ODE_RHS_OSTREAM_ADATPOR_HPP
#define STAN_MATH_TORSTEN_ODE_RHS_OSTREAM_ADATPOR_HPP

#include <stan/math/torsten/dsolve/pmx_user_jacobian.hpp>

namespace torsten {
  template <typename F>
  struct pmx_ode_ostream_adapted {
    using yes = double;
    using no = bool;

    template <typename Functor>
    static double test(decltype(std::declval<Functor&>()(0.0, Eigen::VectorXd(), nullptr,
                                                         std::vector<double>(),
                                                         std::vector<double>(),
                                                         std::vector<int>()))*);

    template <typename Functor>
    static no test(...);

    static constexpr bool value = sizeof(test<F>(nullptr)) == sizeof(yes);
  };
  
  template <typename F>
  struct pmx_ode_ostream_not_adapted {
    using yes = double;
    using no = bool;

    template <typename Functor>
    static double test(decltype(std::declval<Functor&>()(0.0, Eigen::VectorXd(),
                                                         std::vector<double>(),
                                                         std::vector<double>(),
                                                         std::vector<int>(), nullptr))*);

    template <typename Functor>
    static no test(...);

    static constexpr bool value = sizeof(test<F>(nullptr)) == sizeof(yes);
  };

  /** 
   * This adaptor changes the location of ostream arg in functor call so
   * we don't need to make change in stanc compiler for variadic solver
   * backend.
   * 
   */
  template<typename F>
  struct ode_rhs_ostream_adaptor {
    /// whether f is a <code>PMXJacobianPair</code>, see <code>jacobian</code>
    static constexpr bool has_user_jacobian = pmx_has_user_jacobian<F>::value;

    /**
     * A copy, not a reference: the events solver hands the functor to
     * <code>model_factory</code> by value, and the model built there keeps
     * this adaptor beyond that copy's lifetime.
     */
    const F f;

    ode_rhs_ostream_adaptor(F const& f_) : f(f_) {}

    /**
     * The user's Jacobian of the right hand side at doubles, when f pairs
     * the right hand side with it: N x (N + size(parms) + size(x_r)),
     * [df/dx | df/dparms | df/dx_r], where the x_r columns are not read.
     */
    template <typename F0 = F, std::enable_if_t<pmx_has_user_jacobian<F0>::value>* = nullptr>
    Eigen::MatrixXd jacobian(double t, const Eigen::VectorXd& x,
                             std::ostream* pstream,
                             const std::vector<double>& parms,
                             const std::vector<double>& x_r,
                             const std::vector<int>& x_i) const {
      Eigen::MatrixXd jac = dsolve::pmx_call_ode_user_jacobian(f.jac_, t, x, pstream,
                                                               parms, x_r, x_i);
      dsolve::pmx_check_user_jacobian_size("PMX ODE user Jacobian", jac, x.size(),
                                           x.size() + parms.size() + x_r.size());
      return jac;
    }

    template <typename F0, typename T0, typename T1, typename T2,
              std::enable_if_t<pmx_ode_ostream_not_adapted<F0>::value>* = nullptr>
    Eigen::Matrix<stan::return_type_t<T0, T1, T2>, -1, 1>
    impl_(F0 const& f0,
          const T0& t, const Eigen::Matrix<T1, -1, 1> &x,
          std::ostream* pstream,
          const std::vector<T2>& parms,
          const std::vector<double>& x_r, const std::vector<int>& x_i) const {
      return f0(t, x, parms, x_r, x_i, pstream);
    }

    template <typename F0, typename T0, typename T1, typename T2,
              std::enable_if_t<pmx_ode_ostream_adapted<F0>::value>* = nullptr>
    Eigen::Matrix<stan::return_type_t<T0, T1, T2>, -1, 1>
    impl_(F0 const& f0,
          const T0& t, const Eigen::Matrix<T1, -1, 1> &x,
          std::ostream* pstream,
          const std::vector<T2>& parms,
          const std::vector<double>& x_r, const std::vector<int>& x_i) const {
      return f0(t, x, pstream, parms, x_r, x_i);
    }

    template <typename T0, typename T1, typename T2>
    Eigen::Matrix<stan::return_type_t<T0, T1, T2>, -1, 1>
    operator()(const T0& t, const Eigen::Matrix<T1, -1, 1> &x,
               std::ostream* pstream,
               const std::vector<T2>& parms,
               const std::vector<double>& x_r, const std::vector<int>& x_i) const {
      return impl_(f, t, x, pstream, parms, x_r, x_i);
    }
  };

}
#endif
