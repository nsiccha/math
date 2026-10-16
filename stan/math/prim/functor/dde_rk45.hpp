#ifndef STAN_MATH_PRIM_FUNCTOR_DDE_RK45_HPP
#define STAN_MATH_PRIM_FUNCTOR_DDE_RK45_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/prim/functor/apply.hpp>
#include <stan/math/prim/functor/coupled_dde_system.hpp>
#include <stan/math/prim/functor/dde_retard.hpp>
#include <stan/math/prim/functor/ode_store_sensitivities.hpp>
#include <ostream>
#include <tuple>
#include <vector>

namespace stan {
namespace math {

/**
 * Solve the delay differential equation (DDE) initial value problem with
 * constant delays
 *
 *   y'(t) = f(t, y(t), [y(t - tau_1), ..., y(t - tau_D)]),
 *   y(t) = y0 for t <= t0,
 *
 * at a set of times { t1, t2, t3, ... } with the non-stiff Runge-Kutta 45
 * method of Dormand and Prince, adapted from RETARD by E. Hairer and
 * G. Wanner (see <code>internal::dde_retard_integrator</code> and
 * licenses/retard-license.txt).
 *
 * The step size never exceeds the smallest delay, and the integrator steps
 * exactly onto the breakpoints where derivatives of the solution can jump:
 * every multiple t0 + k tau_j up to the last output time, and every mixed
 * combination t0 + sum_j k_j tau_j with total order sum_j k_j <= 5 (see
 * <code>internal::dde_breakpoints</code>). The work therefore grows at least
 * like (max(ts) - t0) / min(tau).
 *
 * Gradients with respect to y0, tau and args are computed with forward
 * sensitivities, which are integrated as further delayed components together
 * with y (see <code>coupled_dde_system</code>).
 *
 * \p f must define an operator() with the signature
 *   template<typename T_t, typename T_y, typename T_y_lag, typename... T_Args>
 *   Eigen::Matrix<stan::return_type_t<T_t, T_y, T_y_lag, T_Args...>,
 *                 Eigen::Dynamic, 1>
 *     operator()(const T_t& t, const T_y& y, const T_y_lag& y_lag,
 *                std::ostream* msgs, const T_Args&... args);
 *
 * t is the time, y is the vector-valued state, y_lag is the N x D matrix
 * whose column j is y(t - tau_j), msgs is a stream for error messages, and
 * args are optional arguments passed to the DDE solve function (which are
 * passed through to \p f without modification).
 *
 * @tparam F Type of DDE right hand side
 * @tparam T_y0 Type of initial condition
 * @tparam T_t0 Type of initial time (arithmetic)
 * @tparam T_ts Type of output times (arithmetic)
 * @tparam T_tau Type of the delay vector
 * @tparam Args Types of pass-through parameters
 *
 * @param function_name Calling function name (for printing debugging messages)
 * @param f Right hand side of the DDE
 * @param y0_arg Initial state, which is also the constant initial history
 * @param t0 Initial time
 * @param ts Times at which to solve the DDE. All values must be strictly
 *   increasing and greater than t0.
 * @param tau_arg Delays; all positive
 * @param relative_tolerance Relative tolerance of the local error
 * @param absolute_tolerance Absolute tolerance of the local error
 * @param max_num_steps Upper limit on the number of integration steps
 *   (accepted or rejected) to take between each output (error if exceeded)
 * @param[in, out] msgs the print stream for warning messages
 * @param args Extra arguments passed unmodified through to DDE right hand side
 * @return Solution to DDE at times \p ts
 */
template <typename F, typename T_y0, typename T_t0, typename T_ts,
          typename T_tau, typename... Args,
          require_eigen_col_vector_t<T_y0>* = nullptr,
          require_eigen_col_vector_t<T_tau>* = nullptr>
inline std::vector<Eigen::Matrix<
    stan::return_type_t<T_y0, T_t0, T_ts, T_tau, Args...>, Eigen::Dynamic, 1>>
dde_rk45_tol_impl(const char* function_name, const F& f, const T_y0& y0_arg,
                  T_t0 t0, const std::vector<T_ts>& ts, const T_tau& tau_arg,
                  double relative_tolerance, double absolute_tolerance,
                  long int max_num_steps,  // NOLINT(runtime/int)
                  std::ostream* msgs, const Args&... args) {
  static_assert(std::is_arithmetic<T_t0>::value,
                "dde_rk45: the initial time must be data");
  static_assert(std::is_arithmetic<T_ts>::value,
                "dde_rk45: the output times must be data");

  using T_y0_t0 = return_type_t<T_y0, T_t0>;

  Eigen::Matrix<T_y0_t0, Eigen::Dynamic, 1> y0
      = y0_arg.template cast<T_y0_t0>();
  ref_type_t<T_tau> tau = tau_arg;

  check_finite(function_name, "initial state", y0);
  check_finite(function_name, "initial time", t0);
  check_finite(function_name, "times", ts);
  check_finite(function_name, "delays", tau);

  std::tuple<ref_type_t<Args>...> args_ref_tuple(args...);

  math::apply(
      [&](const auto&... args_ref) {
        // Code from https://stackoverflow.com/a/17340003
        std::vector<int> unused_temp{
            0,
            (check_finite(function_name, "dde parameters and data", args_ref),
             0)...};
      },
      args_ref_tuple);

  check_nonzero_size(function_name, "initial state", y0);
  check_nonzero_size(function_name, "times", ts);
  check_ordered(function_name, "times", ts);
  check_less(function_name, "initial time", t0, ts[0]);
  check_positive(function_name, "delays", tau);

  check_positive_finite(function_name, "relative_tolerance",
                        relative_tolerance);
  check_positive_finite(function_name, "absolute_tolerance",
                        absolute_tolerance);
  check_positive(function_name, "max_num_steps", max_num_steps);

  using return_t = return_type_t<T_y0, T_t0, T_ts, T_tau, Args...>;
  // creates basic or coupled system by template specializations
  auto&& coupled_system = math::apply(
      [&](const auto&... args_ref) {
        return coupled_dde_system<F, T_y0_t0, ref_type_t<T_tau>,
                                  ref_type_t<Args>...>(f, y0, tau, msgs,
                                                       args_ref...);
      },
      args_ref_tuple);

  std::vector<double> ts_vec(ts.size());
  for (size_t i = 0; i < ts.size(); ++i) {
    ts_vec[i] = ts[i];
  }
  std::vector<double> tau_vec(tau.size());
  for (Eigen::Index j = 0; j < tau.size(); ++j) {
    tau_vec[j] = value_of(tau.coeff(j));
  }

  std::vector<Eigen::Matrix<return_t, Eigen::Dynamic, 1>> y;
  y.reserve(ts.size());

  // the output stores the sensitivities with respect to y0, args and tau in
  // that order, which is the order of the coupled system's parameters
  auto observer = [&](size_t i, const Eigen::VectorXd& coupled_state) -> void {
    std::vector<double> coupled_state_vec(
        coupled_state.data(), coupled_state.data() + coupled_state.size());
    math::apply(
        [&](const auto&... args_ref) {
          y.emplace_back(ode_store_sensitivities(
              f, coupled_state_vec, y0, t0, ts[i], msgs, args_ref..., tau));
        },
        args_ref_tuple);
  };

  internal::dde_retard_integrator<std::decay_t<decltype(coupled_system)>>
      integrator(function_name, coupled_system, tau_vec, relative_tolerance,
                 absolute_tolerance, max_num_steps);
  const Eigen::VectorXd initial_state = coupled_system.initial_state();
  integrator.start(t0, initial_state, initial_state);
  integrator.advance_to(ts_vec.back(), ts_vec, observer);

  return y;
}

/**
 * Solve the DDE initial value problem with constant delays
 *   y'(t) = f(t, y(t), [y(t - tau_1), ..., y(t - tau_D)]),
 *   y(t) = y0 for t <= t0,
 * at a set of times { t1, t2, t3, ... } with the non-stiff Runge-Kutta 45
 * method of Dormand and Prince, adapted from RETARD (see
 * <code>dde_rk45_tol_impl</code>).
 *
 * \p f must define an operator() with the signature
 *   template<typename T_t, typename T_y, typename T_y_lag, typename... T_Args>
 *   Eigen::Matrix<stan::return_type_t<T_t, T_y, T_y_lag, T_Args...>,
 *                 Eigen::Dynamic, 1>
 *     operator()(const T_t& t, const T_y& y, const T_y_lag& y_lag,
 *                std::ostream* msgs, const T_Args&... args);
 *
 * where column j of the N x D matrix y_lag is y(t - tau_j).
 *
 * @tparam F Type of DDE right hand side
 * @tparam T_y0 Type of initial condition
 * @tparam T_t0 Type of initial time (arithmetic)
 * @tparam T_ts Type of output times (arithmetic)
 * @tparam T_tau Type of the delay vector
 * @tparam Args Types of pass-through parameters
 *
 * @param f Right hand side of the DDE
 * @param y0 Initial state, which is also the constant initial history
 * @param t0 Initial time
 * @param ts Times at which to solve the DDE. All values must be strictly
 *   increasing and greater than t0.
 * @param tau Delays; all positive
 * @param relative_tolerance Relative tolerance of the local error
 * @param absolute_tolerance Absolute tolerance of the local error
 * @param max_num_steps Upper limit on the number of integration steps
 *   (accepted or rejected) to take between each output (error if exceeded)
 * @param[in, out] msgs the print stream for warning messages
 * @param args Extra arguments passed unmodified through to DDE right hand side
 * @return Solution to DDE at times \p ts
 */
template <typename F, typename T_y0, typename T_t0, typename T_ts,
          typename T_tau, typename... Args,
          require_eigen_col_vector_t<T_y0>* = nullptr,
          require_eigen_col_vector_t<T_tau>* = nullptr>
inline std::vector<Eigen::Matrix<
    stan::return_type_t<T_y0, T_t0, T_ts, T_tau, Args...>, Eigen::Dynamic, 1>>
dde_rk45_tol(const F& f, const T_y0& y0, T_t0 t0, const std::vector<T_ts>& ts,
             const T_tau& tau, double relative_tolerance,
             double absolute_tolerance,
             long int max_num_steps,  // NOLINT(runtime/int)
             std::ostream* msgs, const Args&... args) {
  return dde_rk45_tol_impl("dde_rk45_tol", f, y0, t0, ts, tau,
                           relative_tolerance, absolute_tolerance,
                           max_num_steps, msgs, args...);
}

/**
 * Solve the DDE initial value problem with constant delays
 *   y'(t) = f(t, y(t), [y(t - tau_1), ..., y(t - tau_D)]),
 *   y(t) = y0 for t <= t0,
 * at a set of times { t1, t2, t3, ... } with the non-stiff Runge-Kutta 45
 * method of Dormand and Prince, adapted from RETARD (see
 * <code>dde_rk45_tol_impl</code>), with the defaults of
 * <code>ode_rk45</code> for relative_tolerance (1e-6), absolute_tolerance
 * (1e-6) and max_num_steps (1e6).
 *
 * \p f must define an operator() with the signature
 *   template<typename T_t, typename T_y, typename T_y_lag, typename... T_Args>
 *   Eigen::Matrix<stan::return_type_t<T_t, T_y, T_y_lag, T_Args...>,
 *                 Eigen::Dynamic, 1>
 *     operator()(const T_t& t, const T_y& y, const T_y_lag& y_lag,
 *                std::ostream* msgs, const T_Args&... args);
 *
 * where column j of the N x D matrix y_lag is y(t - tau_j).
 *
 * @tparam F Type of DDE right hand side
 * @tparam T_y0 Type of initial condition
 * @tparam T_t0 Type of initial time (arithmetic)
 * @tparam T_ts Type of output times (arithmetic)
 * @tparam T_tau Type of the delay vector
 * @tparam Args Types of pass-through parameters
 *
 * @param f Right hand side of the DDE
 * @param y0 Initial state, which is also the constant initial history
 * @param t0 Initial time
 * @param ts Times at which to solve the DDE. All values must be strictly
 *   increasing and greater than t0.
 * @param tau Delays; all positive
 * @param[in, out] msgs the print stream for warning messages
 * @param args Extra arguments passed unmodified through to DDE right hand side
 * @return Solution to DDE at times \p ts
 */
template <typename F, typename T_y0, typename T_t0, typename T_ts,
          typename T_tau, typename... Args,
          require_eigen_col_vector_t<T_y0>* = nullptr,
          require_eigen_col_vector_t<T_tau>* = nullptr>
inline std::vector<Eigen::Matrix<
    stan::return_type_t<T_y0, T_t0, T_ts, T_tau, Args...>, Eigen::Dynamic, 1>>
dde_rk45(const F& f, const T_y0& y0, T_t0 t0, const std::vector<T_ts>& ts,
         const T_tau& tau, std::ostream* msgs, const Args&... args) {
  double relative_tolerance = 1e-6;
  double absolute_tolerance = 1e-6;
  long int max_num_steps = 1e6;  // NOLINT(runtime/int)

  return dde_rk45_tol_impl("dde_rk45", f, y0, t0, ts, tau, relative_tolerance,
                           absolute_tolerance, max_num_steps, msgs, args...);
}

}  // namespace math
}  // namespace stan
#endif
