#ifndef TORSTEN_PMX_TEST_FUNCTORS_HPP
#define TORSTEN_PMX_TEST_FUNCTORS_HPP

#include <stan/math/torsten/torsten.hpp>

#define PMX_ADD_FUNCTOR(name, func)             \
  struct name##_functor {                       \
    template <typename... Ts>                   \
    auto operator()(Ts... args) const {         \
      return func(args...);                     \
    }                                           \
  };

PMX_ADD_FUNCTOR(pmx_solve_onecpt, torsten::pmx_solve_onecpt);
PMX_ADD_FUNCTOR(pmx_solve_twocpt, torsten::pmx_solve_twocpt);
PMX_ADD_FUNCTOR(pmx_solve_linode, torsten::pmx_solve_linode);
PMX_ADD_FUNCTOR(pmx_solve_adams, torsten::pmx_solve_adams);
PMX_ADD_FUNCTOR(pmx_solve_bdf, torsten::pmx_solve_bdf);
PMX_ADD_FUNCTOR(pmx_solve_rk45, torsten::pmx_solve_rk45);
PMX_ADD_FUNCTOR(pmx_solve_onecpt_effcpt, torsten::pmx_solve_onecpt_effcpt);
PMX_ADD_FUNCTOR(pmx_solve_twocpt_effcpt, torsten::pmx_solve_twocpt_effcpt);
PMX_ADD_FUNCTOR(pmx_solve_onecpt_rk45, torsten::pmx_solve_onecpt_rk45);
PMX_ADD_FUNCTOR(pmx_solve_onecpt_bdf, torsten::pmx_solve_onecpt_bdf);
PMX_ADD_FUNCTOR(pmx_solve_twocpt_rk45, torsten::pmx_solve_twocpt_rk45);
PMX_ADD_FUNCTOR(pmx_solve_twocpt_bdf, torsten::pmx_solve_twocpt_bdf);

/**
 * Right hand side of pmx_solve_dde_rk45 made from the right hand side of an
 * ODE in the (t, y, theta, x_r, x_i, msgs) form of PMXOneCptODE: it ignores
 * the lagged states.
 */
template <typename F_ode>
struct pmx_dde_lag_free_rhs {
  F_ode f;

  template <typename T0, typename T1, typename T2, typename T3>
  Eigen::Matrix<stan::return_type_t<T0, T1, T2, T3>, -1, 1>
  operator()(const T0& t, const Eigen::Matrix<T1, -1, 1>& y,
             const Eigen::Matrix<T2, -1, -1>& y_lag,
             const std::vector<T3>& theta, const std::vector<double>& x_r,
             const std::vector<int>& x_i, std::ostream* msgs) const {
    const std::vector<T1> y_vec(y.data(), y.data() + y.size());
    const auto dy_dt = f(t, y_vec, theta, x_r, x_i, msgs);
    Eigen::Matrix<stan::return_type_t<T0, T1, T2, T3>, -1, 1> res(dy_dt.size());
    for (size_t i = 0; i < dy_dt.size(); ++i) {
      res(i) = dy_dt[i];
    }
    return res;
  }
};

/**
 * pmx_solve_dde_rk45 with the lag-free right hand side of an ODE and a zero
 * history, called like pmx_solve_rk45: it solves the same problem. The delay
 * does not change the solution; it limits the step size and adds
 * breakpoints, which exercises the DDE machinery.
 */
struct pmx_solve_dde_rk45_lag_free_functor {
  template <typename F, typename T0, typename T1, typename T2, typename T3,
            typename T_theta, typename... Ts>
  auto operator()(const F& f, int ncmt, const std::vector<T0>& time,
                  const std::vector<T1>& amt, const std::vector<T2>& rate,
                  const std::vector<T3>& ii, const std::vector<int>& evid,
                  const std::vector<int>& cmt, const std::vector<int>& addl,
                  const std::vector<int>& ss, const std::vector<T_theta>& theta,
                  const Ts&... args) const {
    Eigen::VectorXd tau(1);
    tau << 0.7;
    const Eigen::VectorXd y_hist = Eigen::VectorXd::Zero(ncmt);
    return torsten::pmx_solve_dde_rk45(pmx_dde_lag_free_rhs<F>{f}, ncmt, time,
                                       amt, rate, ii, evid, cmt, addl, ss,
                                       theta, tau, y_hist, args...);
  }
};

#endif
