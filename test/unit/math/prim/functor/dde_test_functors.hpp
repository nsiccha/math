#ifndef TEST_UNIT_MATH_PRIM_FUNCTOR_DDE_TEST_FUNCTORS_HPP
#define TEST_UNIT_MATH_PRIM_FUNCTOR_DDE_TEST_FUNCTORS_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/functor/dde_retard.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace stan {
namespace test {
namespace dde {

/**
 * Linear DDE with pure lags: y'(t) = sum_j A_j y(t - tau_j).
 */
struct linear_lag_rhs {
  template <typename T_t, typename T_y, typename T_lag, typename T_A>
  Eigen::Matrix<stan::return_type_t<T_t, T_y, T_lag, T_A>, Eigen::Dynamic, 1>
  operator()(const T_t& t, const T_y& y, const T_lag& y_lag, std::ostream* msgs,
             const std::vector<T_A>& A) const {
    using T_return = stan::return_type_t<T_t, T_y, T_lag, T_A>;
    Eigen::Matrix<T_return, Eigen::Dynamic, 1> dydt
        = Eigen::Matrix<T_return, Eigen::Dynamic, 1>::Zero(y.size());
    for (size_t j = 0; j < A.size(); ++j) {
      dydt += A[j] * y_lag.col(j);
    }
    return dydt;
  }
};

/**
 * Scalar DDE y'(t) = -a y(t - tau).
 */
struct scalar_lag_rhs {
  template <typename T_t, typename T_y, typename T_lag, typename T_a>
  Eigen::Matrix<stan::return_type_t<T_t, T_y, T_lag, T_a>, Eigen::Dynamic, 1>
  operator()(const T_t& t, const T_y& y, const T_lag& y_lag, std::ostream* msgs,
             const T_a& a) const {
    Eigen::Matrix<stan::return_type_t<T_t, T_y, T_lag, T_a>, Eigen::Dynamic, 1>
        dydt(1);
    dydt(0) = -a * y_lag(0, 0);
    return dydt;
  }
};

/**
 * Nonlinear two-state system with two delays, explicit time dependence, a
 * parameter vector and a data argument:
 *   y1' = -k1 y1(t) + r y2(t - tau_1) / (1 + y2(t - tau_1)^2)
 *   y2' = k2 y1(t - tau_2) - k3 y2(t) + c sin(t)
 * with theta = (k1, r, k2, k3).
 */
struct nonlinear_rhs {
  template <typename T_t, typename T_y, typename T_lag, typename T_theta>
  Eigen::Matrix<stan::return_type_t<T_t, T_y, T_lag, T_theta>, Eigen::Dynamic,
                1>
  operator()(const T_t& t, const T_y& y, const T_lag& y_lag, std::ostream* msgs,
             const std::vector<T_theta>& theta, double c) const {
    using std::sin;
    Eigen::Matrix<stan::return_type_t<T_t, T_y, T_lag, T_theta>, Eigen::Dynamic,
                  1>
        dydt(2);
    dydt(0) = -theta[0] * y(0)
              + theta[1] * y_lag(1, 0) / (1 + y_lag(1, 0) * y_lag(1, 0));
    dydt(1) = theta[2] * y_lag(0, 1) - theta[3] * y(1) + c * sin(t);
    return dydt;
  }
};

/**
 * y' = -y(t - tau_1), which checks that y_lag has y.size() rows and
 * expected_delays columns.
 */
struct shape_check_rhs {
  template <typename T_t, typename T_y, typename T_lag>
  Eigen::Matrix<stan::return_type_t<T_t, T_y, T_lag>, Eigen::Dynamic, 1>
  operator()(const T_t& t, const T_y& y, const T_lag& y_lag, std::ostream* msgs,
             int expected_delays) const {
    if (y_lag.rows() != y.size() || y_lag.cols() != expected_delays) {
      throw std::invalid_argument("y_lag has the wrong shape");
    }
    return -y_lag.col(0);
  }
};

/**
 * Returns a derivative vector with one entry too many.
 */
struct wrong_size_rhs {
  template <typename T_t, typename T_y, typename T_lag>
  Eigen::Matrix<stan::return_type_t<T_t, T_y, T_lag>, Eigen::Dynamic, 1>
  operator()(const T_t& t, const T_y& y, const T_lag& y_lag,
             std::ostream* msgs) const {
    return Eigen::Matrix<stan::return_type_t<T_t, T_y, T_lag>, Eigen::Dynamic,
                         1>::Zero(y.size() + 1);
  }
};

/**
 * Returns NaN after t = 0.5.
 */
struct nan_rhs {
  template <typename T_t, typename T_y, typename T_lag>
  Eigen::Matrix<stan::return_type_t<T_t, T_y, T_lag>, Eigen::Dynamic, 1>
  operator()(const T_t& t, const T_y& y, const T_lag& y_lag,
             std::ostream* msgs) const {
    Eigen::Matrix<stan::return_type_t<T_t, T_y, T_lag>, Eigen::Dynamic, 1> dydt
        = -y_lag.col(0);
    if (t > 0.5) {
      dydt(0) = std::numeric_limits<double>::quiet_NaN();
    }
    return dydt;
  }
};

/**
 * Throws a domain error after t = 0.5.
 */
struct throwing_rhs {
  template <typename T_t, typename T_y, typename T_lag>
  Eigen::Matrix<stan::return_type_t<T_t, T_y, T_lag>, Eigen::Dynamic, 1>
  operator()(const T_t& t, const T_y& y, const T_lag& y_lag,
             std::ostream* msgs) const {
    if (t > 0.5) {
      throw std::domain_error("throwing_rhs: thrown by the right hand side");
    }
    return -y_lag.col(0);
  }
};

/**
 * Exact solution of the linear pure-lag DDE
 *   y'(t) = sum_j A_j y(t - m_j delta),  y(t) = y0 for t <= t0,
 * with positive integers m_j, by the method of steps: on
 * [t0 + i delta, t0 + (i + 1) delta] the solution is a polynomial in
 * u = t - t0 - i delta whose coefficients follow from the polynomials of the
 * earlier intervals.
 */
inline Eigen::VectorXd linear_lag_exact(double t, double t0,
                                        const Eigen::VectorXd& y0, double delta,
                                        const std::vector<Eigen::MatrixXd>& A,
                                        const std::vector<int>& m) {
  using poly = std::vector<Eigen::VectorXd>;
  const double s = t - t0;
  if (s <= 0) {
    return y0;
  }
  const int last = std::max(0, static_cast<int>(std::ceil(s / delta)) - 1);
  auto eval = [](const poly& p, double u) {
    Eigen::VectorXd v = Eigen::VectorXd::Zero(p[0].size());
    for (size_t k = p.size(); k-- > 0;) {
      v = v * u + p[k];
    }
    return v;
  };
  std::vector<poly> P;
  for (int i = 0; i <= last; ++i) {
    poly dp;
    for (size_t j = 0; j < A.size(); ++j) {
      const poly lagged = i - m[j] < 0 ? poly{y0} : P[i - m[j]];
      if (dp.size() < lagged.size()) {
        dp.resize(lagged.size(), Eigen::VectorXd::Zero(y0.size()));
      }
      for (size_t k = 0; k < lagged.size(); ++k) {
        dp[k] += A[j] * lagged[k];
      }
    }
    poly p(dp.size() + 1);
    p[0] = i == 0 ? y0 : eval(P[i - 1], delta);
    for (size_t k = 0; k < dp.size(); ++k) {
      p[k + 1] = dp[k] / (k + 1.0);
    }
    P.push_back(p);
  }
  return eval(P[last], s - last * delta);
}

/**
 * Exact solution of y'(t) = -a y(t - tau), y = y0 for t <= t0.
 */
inline double scalar_lag_exact(double t, double t0, double y0, double a,
                               double tau) {
  Eigen::VectorXd y0_vec(1);
  y0_vec << y0;
  Eigen::MatrixXd A(1, 1);
  A << -a;
  return linear_lag_exact(t, t0, y0_vec, tau, {A}, {1})(0);
}

/**
 * Exact solution of y'(t) = -a y(t - tau) with history y = 0 for t < 0 and
 * y(0) = 1 (a unit jump at t = 0), with its derivatives with respect to a and
 * tau. On [n tau, (n + 1) tau],
 *   y(t) = sum_{k=0}^n (-a)^k (t - k tau)^k / k!.
 * dy / dtau jumps from 0 to a at t = tau.
 */
inline void unit_jump_lag_exact(double t, double a, double tau, double& y,
                                double& dy_da, double& dy_dtau) {
  y = 0;
  dy_da = 0;
  dy_dtau = 0;
  double factorial = 1;
  for (int k = 0; k * tau <= t; ++k) {
    if (k > 0) {
      factorial *= k;
    }
    const double u = t - k * tau;
    y += std::pow(-a, k) * std::pow(u, k) / factorial;
    if (k > 0) {
      dy_da += -k * std::pow(-a, k - 1) * std::pow(u, k) / factorial;
      dy_dtau += -k * std::pow(-a, k) * std::pow(u, k - 1) / (factorial / k);
    }
  }
}

/**
 * Two-state system with a saturable and a linear lag of the first state,
 * and a zero-order input:
 *   x1' = -k x1 + rate
 *   x2' = x1(t - tau_1) / (1 + x1(t - tau_1)) + 0.3 x1(t - tau_2) - k2 x2
 * with theta = (k, k2, ...); further entries of theta are not used by the
 * right hand side (they are the parameters of the dosing schedule). The rate
 * is read through a pointer, so that the schedule can change it.
 */
struct dosing_rhs {
  const double* rate;

  template <typename T_t, typename T_y, typename T_lag, typename T_theta>
  Eigen::Matrix<stan::return_type_t<T_t, T_y, T_lag, T_theta>, Eigen::Dynamic,
                1>
  operator()(const T_t& t, const T_y& y, const T_lag& y_lag, std::ostream* msgs,
             const std::vector<T_theta>& theta) const {
    Eigen::Matrix<stan::return_type_t<T_t, T_y, T_lag, T_theta>, Eigen::Dynamic,
                  1>
        dydt(2);
    dydt(0) = -theta[0] * y(0) + *rate;
    dydt(1)
        = y_lag(0, 0) / (1 + y_lag(0, 0)) + 0.3 * y_lag(0, 1) - theta[1] * y(1);
    return dydt;
  }
};

/**
 * Dosing schedule of dosing_rhs, starting at t = 0 from x = 0 with history 0:
 * a bolus into x1 at dose_time, an infusion into x1 on
 * [infusion_start, infusion_start + infusion_duration), and a second bolus
 * into x1 at second_dose_time.
 */
struct dosing_schedule {
  double dose_time = 0.4;
  double dose = 2.0;
  double infusion_start = 1.5;
  double infusion_duration = 0.8;
  double infusion_rate = 2.0;
  double second_dose_time = 3.0;
  double second_dose = 1.0;
};

/**
 * Integrate the dosing schedule with the resumable DDE integrator and return
 * the state at the output times ts.
 *
 * If the system has sensitivities, its parameters must be
 * theta = (k, k2, dose_time, dose, infusion_duration) followed by the delays:
 * the first bolus then also jumps the sensitivity with respect to dose, and
 * the times of the first bolus and of the end of the infusion depend on
 * theta.
 *
 * @param system system of dosing_rhs, whose rate pointer points to rate
 * @param rate the rate of the system, changed by the schedule
 * @param tau delays
 * @param schedule the schedule
 * @param ts output times, none of which is a dosing time
 * @param tolerance relative and absolute tolerance
 */
template <typename System>
std::vector<Eigen::VectorXd> run_dosing_schedule(
    System& system, double& rate, const std::vector<double>& tau,
    const dosing_schedule& schedule, const std::vector<double>& ts,
    double tolerance) {
  const Eigen::Index N = 2;
  const Eigen::Index P = system.size() / N - 1;
  stan::math::internal::dde_retard_integrator<System> integrator(
      "run_dosing_schedule", system, tau, tolerance, tolerance, 1000000);
  const Eigen::VectorXd zero = Eigen::VectorXd::Zero(system.size());
  rate = 0;
  integrator.start(0.0, zero, zero);
  std::vector<Eigen::VectorXd> out;
  size_t next = 0;
  auto advance = [&](double t1) {
    std::vector<double> ts_part;
    while (next < ts.size() && ts[next] <= t1) {
      ts_part.push_back(ts[next++]);
    }
    integrator.advance_to(t1, ts_part, [&](size_t, const Eigen::VectorXd& z) {
      out.push_back(z);
    });
  };
  // d(time) / dtheta for a time equal to theta[p]
  auto time_sensitivity = [&](Eigen::Index p) {
    if (P == 0) {
      return Eigen::VectorXd();
    }
    Eigen::VectorXd dt_dtheta = Eigen::VectorXd::Zero(P);
    dt_dtheta(p) = 1;
    return dt_dtheta;
  };
  advance(schedule.dose_time);
  Eigen::VectorXd dz = zero;
  dz(0) = schedule.dose;
  if (P > 0) {
    dz(N * (1 + 3)) = 1;
  }
  integrator.jump(dz, time_sensitivity(2));
  advance(schedule.infusion_start);
  integrator.change_rhs(Eigen::VectorXd(),
                        [&] { rate = schedule.infusion_rate; });
  advance(schedule.infusion_start + schedule.infusion_duration);
  integrator.change_rhs(time_sensitivity(4), [&] { rate = 0; });
  advance(schedule.second_dose_time);
  dz = zero;
  dz(0) = schedule.second_dose;
  integrator.jump(dz, Eigen::VectorXd());
  advance(ts.back());
  return out;
}

}  // namespace dde
}  // namespace test
}  // namespace stan
#endif
