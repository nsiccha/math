#ifndef TEST_UNIT_MATH_PRIM_FUNCTOR_DDE_TEST_FUNCTORS_HPP
#define TEST_UNIT_MATH_PRIM_FUNCTOR_DDE_TEST_FUNCTORS_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
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

}  // namespace dde
}  // namespace test
}  // namespace stan
#endif
