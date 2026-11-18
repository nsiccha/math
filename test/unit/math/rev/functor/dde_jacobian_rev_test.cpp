#include <stan/math/rev.hpp>
#include <test/unit/math/rev/util.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

using stan::math::var;

/**
 * Two states, two delays, nonlinear in the lagged states:
 *
 * y0' = -theta0 y0 + theta1 y1(t - tau0) / (1 + y0(t - tau1)^2)
 * y1' = theta0 y0 - k y1 y1(t - tau1)
 *
 * The variadic arguments are theta, an unused real array, an unused
 * integer array and k.
 */
struct lag_rhs {
  template <typename T_t, typename T_y, typename T_lag, typename T_theta,
            typename T_k>
  Eigen::Matrix<stan::return_type_t<T_t, T_y, T_lag, T_theta, T_k>,
                Eigen::Dynamic, 1>
  operator()(const T_t& t, const T_y& y, const T_lag& y_lag, std::ostream* msgs,
             const T_theta& theta, const std::vector<double>& x_r,
             const std::vector<int>& x_i, const T_k& k) const {
    Eigen::Matrix<stan::return_type_t<T_t, T_y, T_lag, T_theta, T_k>,
                  Eigen::Dynamic, 1>
        dy(2);
    dy(0) = -theta(0) * y(0)
            + theta(1) * y_lag(1, 0) / (1 + y_lag(0, 1) * y_lag(0, 1));
    dy(1) = theta(0) * y(0) - k * y(1) * y_lag(1, 1);
    return dy;
  }
};

/**
 * Jacobian of lag_rhs: columns y0, y1, y_lag(0, 0), y_lag(1, 0),
 * y_lag(0, 1), y_lag(1, 1), theta0, theta1, x_r0, k.
 */
struct lag_jac {
  template <typename T_y, typename T_lag, typename T_theta, typename T_k>
  Eigen::MatrixXd operator()(double t, const T_y& y, const T_lag& y_lag,
                             std::ostream* msgs, const T_theta& theta,
                             const std::vector<double>& x_r,
                             const std::vector<int>& x_i, const T_k& k) const {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2, 10);
    const double d = 1 + y_lag(0, 1) * y_lag(0, 1);
    J(0, 0) = -theta(0);
    J(0, 3) = theta(1) / d;
    J(0, 4) = -theta(1) * y_lag(1, 0) * 2 * y_lag(0, 1) / (d * d);
    J(0, 6) = -y(0);
    J(0, 7) = y_lag(1, 0) / d;
    J(1, 0) = theta(0);
    J(1, 1) = -k * y_lag(1, 1);
    J(1, 5) = -k * y(1);
    J(1, 6) = y(0);
    J(1, 9) = -y(1) * y_lag(1, 1);
    return J;
  }
};

/** Right with respect to y and y_lag, zero with respect to the arguments. */
struct lag_state_only_jac {
  template <typename... Args>
  Eigen::MatrixXd operator()(Args&&... args) const {
    Eigen::MatrixXd J = lag_jac()(args...);
    J.rightCols(4).setZero();
    return J;
  }
};

/** A Jacobian without the lag blocks. */
struct lag_short_jac {
  template <typename... Args>
  Eigen::MatrixXd operator()(Args&&... args) const {
    Eigen::MatrixXd J = lag_jac()(args...);
    Eigen::MatrixXd J_short(2, 6);
    J_short << J.leftCols(2), J.rightCols(4);
    return J_short;
  }
};

const std::vector<double> x_r{2.0};
const std::vector<int> x_i{1};
const std::vector<double> ts{1.0, 2.0, 3.5, 5.0};

/**
 * Values and gradients with respect to (y0, theta, k, tau) of every
 * output, through the autodiff path or the user-Jacobian path.
 */
template <bool UseJac, typename FJac = lag_jac>
std::vector<std::pair<double, std::vector<double>>> solve(bool y0_var,
                                                          bool theta_var,
                                                          bool k_var,
                                                          bool tau_var) {
  Eigen::Matrix<var, Eigen::Dynamic, 1> y0(2), theta(2), tau(2);
  y0 << 1.0, 0.5;
  theta << 0.8, 0.5;
  tau << 0.7, 1.3;
  var k = 0.3;
  std::vector<var> inputs{y0(0), y0(1), theta(0), theta(1), k, tau(0), tau(1)};
  std::vector<std::pair<double, std::vector<double>>> out;
  auto run = [&](const auto& y0_a, const auto& theta_a, const auto& k_a,
                 const auto& tau_a) {
    const auto ys = [&]() {
      if constexpr (UseJac) {
        return stan::math::dde_rk45_tol(lag_rhs(), FJac(), y0_a, 0.0, ts, tau_a,
                                        1e-10, 1e-10, 1000000, nullptr, theta_a,
                                        x_r, x_i, k_a);
      } else {
        return stan::math::dde_rk45_tol(lag_rhs(), y0_a, 0.0, ts, tau_a, 1e-10,
                                        1e-10, 1000000, nullptr, theta_a, x_r,
                                        x_i, k_a);
      }
    }();
    if constexpr (stan::is_var<stan::value_type_t<
                      std::decay_t<decltype(ys[0])>>>::value) {
      for (const auto& y : ys) {
        for (int n = 0; n < 2; ++n) {
          stan::math::set_zero_all_adjoints();
          stan::math::grad(y(n).vi_);
          std::vector<double> g;
          for (const var& x : inputs) {
            g.push_back(x.adj());
          }
          out.emplace_back(y(n).val(), g);
        }
      }
    }
  };
  const Eigen::VectorXd y0_d = stan::math::value_of(y0);
  const Eigen::VectorXd theta_d = stan::math::value_of(theta);
  const Eigen::VectorXd tau_d = stan::math::value_of(tau);
  const double k_d = k.val();
  auto with_tau = [&](const auto& y0_a, const auto& theta_a, const auto& k_a) {
    if (tau_var) {
      run(y0_a, theta_a, k_a, tau);
    } else {
      run(y0_a, theta_a, k_a, tau_d);
    }
  };
  auto with_k = [&](const auto& y0_a, const auto& theta_a) {
    if (k_var) {
      with_tau(y0_a, theta_a, k);
    } else {
      with_tau(y0_a, theta_a, k_d);
    }
  };
  auto with_theta = [&](const auto& y0_a) {
    if (theta_var) {
      with_k(y0_a, theta);
    } else {
      with_k(y0_a, theta_d);
    }
  };
  if (y0_var) {
    with_theta(y0);
  } else {
    with_theta(y0_d);
  }
  stan::math::recover_memory();
  return out;
}

}  // namespace

/**
 * The dde_rk45 overloads taking a Jacobian reproduce the values and
 * gradients of the autodiff path for every combination of autodiff and
 * data initial state, parameters and delays.
 */
TEST(dde_jacobian, matches_autodiff_path) {
  for (int mask = 1; mask < 16; ++mask) {
    const bool y0_var = mask & 1;
    const bool theta_var = mask & 2;
    const bool k_var = mask & 4;
    const bool tau_var = mask & 8;
    const auto ad = solve<false>(y0_var, theta_var, k_var, tau_var);
    const auto jac = solve<true>(y0_var, theta_var, k_var, tau_var);
    ASSERT_EQ(ad.size(), jac.size());
    for (size_t i = 0; i < ad.size(); ++i) {
      EXPECT_NEAR(ad[i].first, jac[i].first,
                  1e-8 * std::max(1.0, std::fabs(ad[i].first)))
          << "mask " << mask << ", output " << i;
      for (size_t j = 0; j < ad[i].second.size(); ++j) {
        EXPECT_NEAR(ad[i].second[j], jac[i].second[j],
                    1e-8 * std::max(1.0, std::fabs(ad[i].second[j])))
            << "mask " << mask << ", output " << i << ", input " << j;
      }
    }
  }
}

/**
 * The solver uses the user Jacobian: one that is zero with respect to the
 * arguments gives zero gradients with respect to theta and k.
 */
TEST(dde_jacobian, uses_user_jacobian) {
  const auto ad = solve<false>(true, true, true, false);
  const auto state_only
      = solve<true, lag_state_only_jac>(true, true, true, false);
  ASSERT_EQ(ad.size(), state_only.size());
  for (size_t i = 0; i < ad.size(); ++i) {
    EXPECT_NEAR(ad[i].first, state_only[i].first, 1e-8);
    for (size_t j = 0; j < 2; ++j) {
      EXPECT_NEAR(ad[i].second[j], state_only[i].second[j], 1e-8);
    }
    for (size_t j = 2; j < 5; ++j) {
      EXPECT_EQ(state_only[i].second[j], 0.0) << "output " << i;
    }
  }
}

/** A Jacobian of the wrong size is an error. */
TEST(dde_jacobian, wrong_size_throws) {
  EXPECT_THROW((solve<true, lag_short_jac>(false, true, false, false)),
               std::invalid_argument);
  stan::math::recover_memory();
}

/** dde_jacobian_error: round-off for the right Jacobian, large otherwise. */
TEST(dde_jacobian, dde_jacobian_error) {
  Eigen::VectorXd y(2), theta(2);
  y << 1.0, 0.5;
  theta << 0.8, 0.5;
  Eigen::MatrixXd y_lag(2, 2);
  y_lag << 0.9, 1.1, 0.4, 0.6;
  EXPECT_LT(stan::math::dde_jacobian_error(lag_rhs(), lag_jac(), 1.0, y, y_lag,
                                           nullptr, theta, x_r, x_i, 0.3),
            1e-14);
  EXPECT_GT(
      stan::math::dde_jacobian_error(lag_rhs(), lag_state_only_jac(), 1.0, y,
                                     y_lag, nullptr, theta, x_r, x_i, 0.3),
      0.1);
}
