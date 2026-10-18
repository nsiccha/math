#ifndef STAN_MATH_TORSTEN_PMX_SOLVE_DDE_RK45_HPP
#define STAN_MATH_TORSTEN_PMX_SOLVE_DDE_RK45_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/core/precomputed_gradients.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/torsten/pmx_check.hpp>
#include <stan/math/torsten/to_array_2d.hpp>
#include <stan/math/torsten/pmx_dde_schedule.hpp>
#include <stan/math/torsten/pmx_dde_solver.hpp>
#include <Eigen/Dense>
#include <cstdlib>
#include <ostream>
#include <string>
#include <vector>

namespace torsten {

  namespace internal {

    inline PMXDdeControls pmx_dde_controls(std::ostream* msgs) {
      PMXDdeControls c;
      c.msgs = msgs;
      return c;
    }

    inline PMXDdeControls pmx_dde_controls(double rtol, double atol,
                                           double max_num_steps,
                                           std::ostream* msgs) {
      PMXDdeControls c = pmx_dde_controls(msgs);
      c.rtol = rtol;
      c.atol = atol;
      c.max_num_steps = static_cast<long int>(max_num_steps);  // NOLINT(runtime/int)
      return c;
    }

    inline PMXDdeControls pmx_dde_controls(double rtol, double atol,
                                           double max_num_steps,
                                           double as_rtol, double as_atol,
                                           double as_max_num_steps,
                                           std::ostream* msgs) {
      PMXDdeControls c = pmx_dde_controls(rtol, atol, max_num_steps, msgs);
      c.as_rtol = as_rtol;
      c.as_atol = as_atol;
      c.as_max_num_steps = static_cast<long int>(as_max_num_steps);  // NOLINT(runtime/int)
      return c;
    }

    /**
     * The result matrix: values, with the forward sensitivities as
     * precomputed gradients if any input is an autodiff variable.
     */
    template <typename T>
    Eigen::Matrix<T, -1, -1>
    pmx_dde_result(const Eigen::MatrixXd& values,
                   const std::vector<Eigen::MatrixXd>& sensitivities,
                   const PMXDdeParameters& params);

    template <>
    inline Eigen::MatrixXd
    pmx_dde_result<double>(const Eigen::MatrixXd& values,
                           const std::vector<Eigen::MatrixXd>& sensitivities,
                           const PMXDdeParameters& params) {
      return values;
    }

    template <>
    inline Eigen::Matrix<stan::math::var, -1, -1>
    pmx_dde_result<stan::math::var>(const Eigen::MatrixXd& values,
                                    const std::vector<Eigen::MatrixXd>& sensitivities,
                                    const PMXDdeParameters& params) {
      const std::vector<stan::math::var> operands = params.operands();
      Eigen::Matrix<stan::math::var, -1, -1> res(values.rows(), values.cols());
      std::vector<double> gradients(params.size());
      for (Eigen::Index k = 0; k < values.cols(); ++k) {
        for (Eigen::Index i = 0; i < values.rows(); ++i) {
          for (size_t p = 0; p < params.size(); ++p) {
            gradients[p] = sensitivities[k].coeff(i, p);
          }
          res.coeffRef(i, k) = stan::math::precomputed_gradients(
              values.coeff(i, k), operands, gradients);
        }
      }
      return res;
    }

    /**
     * Check that a 2-d array of per-event data has one row, or one per event.
     */
    template <typename T>
    void pmx_dde_check_rows(const char* function, const char* name,
                            const std::vector<std::vector<T> >& x, size_t n) {
      if (x.size() != n && x.size() != 1) {
        stan::math::invalid_argument(function, name, x.size(), "has ",
                                     " rows, but must have 1 or one per event");
      }
    }

    /**
     * Implementation of <code>pmx_solve_dde_rk45</code> with all optional
     * arguments given, as 2-d arrays.
     */
    template <typename F, typename T0, typename T1, typename T2, typename T3,
              typename T4, typename T_tau, typename T_hist, typename T5,
              typename T6>
    stan::matrix_return_t<T0, T1, T2, T3, T4, T_tau, T_hist, T5, T6>
    pmx_solve_dde_rk45(const F& f, int nCmt,
                       const std::vector<T0>& time,
                       const std::vector<T1>& amt,
                       const std::vector<T2>& rate,
                       const std::vector<T3>& ii,
                       const std::vector<int>& evid,
                       const std::vector<int>& cmt,
                       const std::vector<int>& addl,
                       const std::vector<int>& ss,
                       const std::vector<std::vector<T4> >& theta,
                       const Eigen::Matrix<T_tau, -1, 1>& tau,
                       const Eigen::Matrix<T_hist, -1, 1>& y_hist,
                       const std::vector<std::vector<T5> >& biovar,
                       const std::vector<std::vector<T6> >& tlag,
                       const std::vector<std::vector<double> >& x_r,
                       const std::vector<std::vector<int> >& x_i,
                       const PMXDdeControls& controls) {
      using stan::math::value_of;
      using T_return = stan::return_type_t<T0, T1, T2, T3, T4, T_tau, T_hist, T5, T6>;
      static const char* function = PMXDdeSolver<F>::function_name;

      stan::math::check_positive(function, "nCmt", nCmt);
      pmx_check(time, amt, rate, ii, evid, cmt, addl, ss, theta, function);
      pmx_check(time.size(), nCmt, biovar, tlag);
      pmx_dde_check_rows(function, "x_r", x_r, time.size());
      pmx_dde_check_rows(function, "x_i", x_i, time.size());
      for (size_t i = 0; i < time.size(); ++i) {
        if (evid[i] == 1 || evid[i] == 4 || evid[i] == 8) {
          stan::math::check_bounded(function, "dosing compartment", cmt[i], 1, nCmt);
        } else if (evid[i] == 2 && cmt[i] < 0) {
          stan::math::check_bounded(function, "turned-off compartment", -cmt[i], 1, nCmt);
        }
      }
      stan::math::check_positive_finite(function, "tau", tau);
      stan::math::check_size_match(function, "y_hist", y_hist.size(), "nCmt", nCmt);
      stan::math::check_finite(function, "y_hist", y_hist);
      stan::math::check_positive_finite(function, "relative tolerance", controls.rtol);
      stan::math::check_positive_finite(function, "absolute tolerance", controls.atol);
      stan::math::check_positive(function, "max_num_steps", controls.max_num_steps);
      stan::math::check_positive_finite(function, "steady-state relative tolerance",
                                        controls.as_rtol);
      stan::math::check_positive_finite(function, "steady-state absolute tolerance",
                                        controls.as_atol);
      stan::math::check_positive(function, "steady-state max_num_steps",
                                 controls.as_max_num_steps);

      PMXDdeParameters params;
      params.add(theta);
      params.add(tau);
      params.add(y_hist);
      params.add(biovar);
      params.add(tlag);
      params.add(time);
      params.add(amt);
      params.add(rate);
      params.add(ii);
      const size_t P = params.size();

      const std::vector<PMXDdeEvent> events
          = pmx_dde_events(nCmt, time, amt, rate, ii, evid, cmt, addl, ss,
                           theta, biovar, tlag, x_r, x_i, params);

      std::vector<double> tau_d(tau.size());
      Eigen::MatrixXd delay_sensitivities = Eigen::MatrixXd::Zero(P, tau.size());
      for (Eigen::Index j = 0; j < tau.size(); ++j) {
        tau_d[j] = value_of(tau.coeff(j));
        const int p = params.index(tau.coeff(j));
        if (p >= 0) {
          delay_sensitivities.coeffRef(p, j) = 1;
        }
      }
      const Eigen::VectorXd y_hist_d = value_of(y_hist);
      Eigen::MatrixXd y_hist_sensitivities = Eigen::MatrixXd::Zero(nCmt, P);
      for (int i = 0; i < nCmt; ++i) {
        const int p = params.index(y_hist.coeff(i));
        if (p >= 0) {
          y_hist_sensitivities.coeffRef(i, p) = 1;
        }
      }

      PMXDdeSolver<F> solver(f, nCmt, tau_d, delay_sensitivities, y_hist_d,
                             y_hist_sensitivities, controls);
      Eigen::MatrixXd values;
      std::vector<Eigen::MatrixXd> sensitivities;
      solver.solve(events, values, sensitivities);
      return pmx_dde_result<T_return>(values, sensitivities, params);
    }

  }

  /**
   * Solve a PMX model defined by a system of delay differential equations
   * with constant delays, for the events of one subject in NONMEM's format,
   * with the explicit Runge-Kutta method of Dormand and Prince adapted to
   * DDEs (RETARD by Hairer and Wanner).
   *
   * The right hand side is
   *   y'(t) = f(t, y(t), y_lag(t), theta, x_r, x_i) + rate(t),
   * where column j of the nCmt x D matrix y_lag is y(t - tau[j]) and rate is
   * the sum of the infusions running at t. Before the first event, and
   * after a reset (EVID 3 and 4), the state and its history are the constant
   * y_hist. Bolus doses are jumps of the state, which the integrator lands
   * on; ADDL/II, infusions, lag times and bioavailability are handled by
   * Torsten's event machinery exactly as for <code>pmx_solve_rk45</code>.
   *
   * Steady-state dosing (SS = 1) restarts from the steady state of the
   * dose's regimen. With II > 0 it is found by burn-in: the regimen is
   * integrated from y_hist until the states at successive doses agree
   * within as_rtol / as_atol, for at most as_max_num_steps dosing periods
   * (an error otherwise), and the history before the dose is that of the
   * last periods. With II = 0 (a constant infusion, AMT = 0) it is the
   * equilibrium f(t, x, [x, ..., x]) + rate = 0, solved by Powell's method
   * from y_hist. SS = 2 (adding a steady state to the current state) is
   * refused: it relies on superposition, which nonlinear systems, and the
   * history of a DDE, do not have.
   *
   * Gradients are available with respect to every input
   * <code>pmx_solve_rk45</code> accepts as a parameter, and to tau and
   * y_hist. They come from forward sensitivities with respect to the
   * distinct autodiff variables of all inputs, integrated through the
   * history, with the jump conditions of discontinuities whose times depend
   * on the parameters.
   *
   * Signatures, with [] marking optional trailing groups:
   *
   *   pmx_solve_dde_rk45(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss,
   *                      theta, tau, y_hist,
   *                      [biovar, [tlag, [x_r, [x_i]]]],
   *                      [rtol, atol, max_num_steps,
   *                       [as_rtol, as_atol, as_max_num_steps]], msgs)
   *
   * theta, biovar and tlag are 1-d arrays (one row for all events) or 2-d
   * arrays with one row per event; x_r and x_i are 2-d. The default
   * tolerances are 1e-6 and the default step limits 1e6 and 100.
   *
   * @return nCmt x (number of events) matrix of the amounts after each event
   */
  template <typename F, typename T0, typename T1, typename T2, typename T3,
            typename T_theta, typename T_tau, typename T_hist, typename T_biovar,
            typename T_tlag, typename... Ctrl,
            stan::require_all_eigen_col_vector_t<T_tau, T_hist>* = nullptr,
            stan::require_all_not_std_vector_t<Ctrl...>* = nullptr>
  auto pmx_solve_dde_rk45(const F& f, const int nCmt,
                          const std::vector<T0>& time,
                          const std::vector<T1>& amt,
                          const std::vector<T2>& rate,
                          const std::vector<T3>& ii,
                          const std::vector<int>& evid,
                          const std::vector<int>& cmt,
                          const std::vector<int>& addl,
                          const std::vector<int>& ss,
                          const std::vector<T_theta>& theta,
                          const T_tau& tau, const T_hist& y_hist,
                          const std::vector<T_biovar>& biovar,
                          const std::vector<T_tlag>& tlag,
                          const std::vector<std::vector<double> >& x_r,
                          const std::vector<std::vector<int> >& x_i,
                          const Ctrl&... ctrl) {
    using tau_t = Eigen::Matrix<stan::value_type_t<T_tau>, -1, 1>;
    using hist_t = Eigen::Matrix<stan::value_type_t<T_hist>, -1, 1>;
    return internal::pmx_solve_dde_rk45(f, nCmt, time, amt, rate, ii, evid, cmt,
                                        addl, ss, to_array_2d(theta),
                                        tau_t(tau), hist_t(y_hist),
                                        to_array_2d(biovar), to_array_2d(tlag),
                                        x_r, x_i,
                                        internal::pmx_dde_controls(ctrl...));
  }

  template <typename F, typename T0, typename T1, typename T2, typename T3,
            typename T_theta, typename T_tau, typename T_hist, typename T_biovar,
            typename T_tlag, typename... Ctrl,
            stan::require_all_eigen_col_vector_t<T_tau, T_hist>* = nullptr,
            stan::require_all_not_std_vector_t<Ctrl...>* = nullptr>
  auto pmx_solve_dde_rk45(const F& f, const int nCmt,
                          const std::vector<T0>& time,
                          const std::vector<T1>& amt,
                          const std::vector<T2>& rate,
                          const std::vector<T3>& ii,
                          const std::vector<int>& evid,
                          const std::vector<int>& cmt,
                          const std::vector<int>& addl,
                          const std::vector<int>& ss,
                          const std::vector<T_theta>& theta,
                          const T_tau& tau, const T_hist& y_hist,
                          const std::vector<T_biovar>& biovar,
                          const std::vector<T_tlag>& tlag,
                          const std::vector<std::vector<double> >& x_r,
                          const Ctrl&... ctrl) {
    return pmx_solve_dde_rk45(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss,
                              theta, tau, y_hist, biovar, tlag, x_r,
                              std::vector<std::vector<int> >{std::vector<int>()},
                              ctrl...);
  }

  template <typename F, typename T0, typename T1, typename T2, typename T3,
            typename T_theta, typename T_tau, typename T_hist, typename T_biovar,
            typename T_tlag, typename... Ctrl,
            stan::require_all_eigen_col_vector_t<T_tau, T_hist>* = nullptr,
            stan::require_all_not_std_vector_t<Ctrl...>* = nullptr>
  auto pmx_solve_dde_rk45(const F& f, const int nCmt,
                          const std::vector<T0>& time,
                          const std::vector<T1>& amt,
                          const std::vector<T2>& rate,
                          const std::vector<T3>& ii,
                          const std::vector<int>& evid,
                          const std::vector<int>& cmt,
                          const std::vector<int>& addl,
                          const std::vector<int>& ss,
                          const std::vector<T_theta>& theta,
                          const T_tau& tau, const T_hist& y_hist,
                          const std::vector<T_biovar>& biovar,
                          const std::vector<T_tlag>& tlag,
                          const Ctrl&... ctrl) {
    return pmx_solve_dde_rk45(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss,
                              theta, tau, y_hist, biovar, tlag,
                              std::vector<std::vector<double> >{std::vector<double>()},
                              ctrl...);
  }

  template <typename F, typename T0, typename T1, typename T2, typename T3,
            typename T_theta, typename T_tau, typename T_hist, typename T_biovar,
            typename... Ctrl,
            stan::require_all_eigen_col_vector_t<T_tau, T_hist>* = nullptr,
            stan::require_all_not_std_vector_t<Ctrl...>* = nullptr>
  auto pmx_solve_dde_rk45(const F& f, const int nCmt,
                          const std::vector<T0>& time,
                          const std::vector<T1>& amt,
                          const std::vector<T2>& rate,
                          const std::vector<T3>& ii,
                          const std::vector<int>& evid,
                          const std::vector<int>& cmt,
                          const std::vector<int>& addl,
                          const std::vector<int>& ss,
                          const std::vector<T_theta>& theta,
                          const T_tau& tau, const T_hist& y_hist,
                          const std::vector<T_biovar>& biovar,
                          const Ctrl&... ctrl) {
    return pmx_solve_dde_rk45(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss,
                              theta, tau, y_hist, biovar,
                              std::vector<std::vector<double> >{std::vector<double>(nCmt, 0.0)},
                              ctrl...);
  }

  template <typename F, typename T0, typename T1, typename T2, typename T3,
            typename T_theta, typename T_tau, typename T_hist, typename... Ctrl,
            stan::require_all_eigen_col_vector_t<T_tau, T_hist>* = nullptr,
            stan::require_all_not_std_vector_t<Ctrl...>* = nullptr>
  auto pmx_solve_dde_rk45(const F& f, const int nCmt,
                          const std::vector<T0>& time,
                          const std::vector<T1>& amt,
                          const std::vector<T2>& rate,
                          const std::vector<T3>& ii,
                          const std::vector<int>& evid,
                          const std::vector<int>& cmt,
                          const std::vector<int>& addl,
                          const std::vector<int>& ss,
                          const std::vector<T_theta>& theta,
                          const T_tau& tau, const T_hist& y_hist,
                          const Ctrl&... ctrl) {
    return pmx_solve_dde_rk45(f, nCmt, time, amt, rate, ii, evid, cmt, addl, ss,
                              theta, tau, y_hist,
                              std::vector<std::vector<double> >{std::vector<double>(nCmt, 1.0)},
                              ctrl...);
  }

}

#endif
