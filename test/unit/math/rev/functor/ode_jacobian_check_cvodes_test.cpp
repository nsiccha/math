// The solvers check every user Jacobian evaluation against autodiff when
// STAN_MATH_CHECK_USER_JACOBIAN is defined.
#define STAN_MATH_CHECK_USER_JACOBIAN
#include <stan/math/rev.hpp>
#include <test/unit/math/rev/util.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace {

using stan::math::var;

// y' = -theta0 y^2 + theta1
struct ode_rhs {
  template <typename T_t, typename T_y, typename T_theta>
  auto operator()(const T_t& t, const T_y& y, std::ostream* msgs,
                  const T_theta& theta) const {
    Eigen::Matrix<stan::return_type_t<T_t, T_y, T_theta>, -1, 1> dy(1);
    dy(0) = -theta[0] * y(0) * y(0) + theta[1];
    return dy;
  }
};

template <bool Right>
struct ode_jac {
  template <typename T_t, typename T_y, typename T_theta>
  Eigen::MatrixXd operator()(const T_t& t, const T_y& y, std::ostream* msgs,
                             const T_theta& theta) const {
    Eigen::MatrixXd J(1, 3);
    J << (Right ? -2 : -1) * theta[0] * y(0), -y(0) * y(0), 1;
    return J;
  }
};

// yp0 + p0 yy0 = 0, yy1 - 2 yy0 = 0
struct dae_res {
  template <typename T_yy, typename T_yp, typename T_p>
  auto operator()(double t, const T_yy& yy, const T_yp& yp, std::ostream* msgs,
                  const std::vector<T_p>& p) const {
    Eigen::Matrix<stan::return_type_t<T_yy, T_yp, T_p>, -1, 1> r(2);
    r(0) = yp(0) + p[0] * yy(0);
    r(1) = yy(1) - 2 * yy(0);
    return r;
  }
};

template <bool Right>
struct dae_jac {
  template <typename T_yy, typename T_yp, typename T_p>
  Eigen::MatrixXd operator()(double t, const T_yy& yy, const T_yp& yp,
                             std::ostream* msgs,
                             const std::vector<T_p>& p) const {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2, 5);
    J(0, 0) = (Right ? 1 : 2) * p[0];
    J(0, 2) = 1;
    J(0, 4) = yy(0);
    J(1, 0) = -2;
    J(1, 1) = 1;
    return J;
  }
};

// y' = -a y(t - tau)
struct dde_rhs {
  template <typename T_t, typename T_y, typename T_lag, typename T_a>
  auto operator()(const T_t& t, const T_y& y, const T_lag& y_lag,
                  std::ostream* msgs, const T_a& a) const {
    Eigen::Matrix<stan::return_type_t<T_t, T_y, T_lag, T_a>, -1, 1> dy(1);
    dy(0) = -a * y_lag(0, 0);
    return dy;
  }
};

template <bool Right>
struct dde_jac {
  template <typename T_t, typename T_y, typename T_lag, typename T_a>
  Eigen::MatrixXd operator()(const T_t& t, const T_y& y, const T_lag& y_lag,
                             std::ostream* msgs, const T_a& a) const {
    Eigen::MatrixXd J(1, 3);
    J << 0, (Right ? -1 : -2) * a, -y_lag(0, 0);
    return J;
  }
};

template <bool Right>
struct ode_pull {
  template <typename T_t, typename T_y, typename T_theta>
  Eigen::VectorXd operator()(const T_t& t, const T_y& y,
                             const Eigen::VectorXd& lambda, std::ostream* msgs,
                             const T_theta& theta) const {
    Eigen::VectorXd g = ode_jac<true>()(t, y, msgs, theta).transpose() * lambda;
    if (!Right) {
      g(1) *= 2;
    }
    return g;
  }
};

const std::vector<double> ts{0.5, 1.0, 2.0};

template <bool Right>
void solve_adjoint_pull() {
  Eigen::VectorXd y0(1);
  y0 << 0.5;
  std::vector<var> theta{1.5, 0.3};
  const Eigen::VectorXd atol = Eigen::VectorXd::Constant(1, 1e-8);
  auto ys = stan::math::ode_adjoint_tol_ctl(
      ode_rhs(), ode_jac<true>(), ode_pull<Right>(), y0, 0.0, ts, 1e-8, atol,
      1e-8, atol, 1e-8, 1e-8, 100000, 150, 1, 2, 2, nullptr, theta);
  // seed every output so that the backward solve runs with lambda != 0
  var lp = 0;
  for (const auto& y : ys) {
    lp += y(0);
  }
  lp.grad();
}

template <bool Right>
void solve_all() {
  Eigen::VectorXd y0(1);
  y0 << 0.5;
  std::vector<var> theta{1.5, 0.3};
  stan::math::ode_rk45(ode_rhs(), ode_jac<Right>(), y0, 0.0, ts, nullptr,
                       theta);
  stan::math::ode_bdf(ode_rhs(), ode_jac<Right>(), y0, 0.0, ts, nullptr, theta);
  Eigen::VectorXd yy0(2), yp0(2);
  yy0 << 1.0, 2.0;
  yp0 << -0.5, -1.0;
  std::vector<var> p{0.5};
  stan::math::dae(dae_res(), dae_jac<Right>(), yy0, yp0, 0.0, ts, nullptr, p);
  Eigen::VectorXd tau(1);
  tau << 0.4;
  var a = 0.7;
  stan::math::dde_rk45(dde_rhs(), dde_jac<Right>(), y0, 0.0, ts, tau, nullptr,
                       a);
}

}  // namespace

TEST(user_jacobian_check, right_jacobians_pass) {
  EXPECT_NO_THROW(solve_all<true>());
  stan::math::recover_memory();
}

TEST(user_jacobian_check, wrong_jacobians_throw) {
  Eigen::VectorXd y0(1);
  y0 << 0.5;
  std::vector<var> theta{1.5, 0.3};
  EXPECT_THROW(stan::math::ode_rk45(ode_rhs(), ode_jac<false>(), y0, 0.0, ts,
                                    nullptr, theta),
               std::domain_error);
  EXPECT_THROW(stan::math::ode_bdf(ode_rhs(), ode_jac<false>(), y0, 0.0, ts,
                                   nullptr, theta),
               std::domain_error);
  Eigen::VectorXd yy0(2), yp0(2);
  yy0 << 1.0, 2.0;
  yp0 << -0.5, -1.0;
  std::vector<var> p{0.5};
  EXPECT_THROW(stan::math::dae(dae_res(), dae_jac<false>(), yy0, yp0, 0.0, ts,
                               nullptr, p),
               std::domain_error);
  Eigen::VectorXd tau(1);
  tau << 0.4;
  var a = 0.7;
  EXPECT_THROW(stan::math::dde_rk45(dde_rhs(), dde_jac<false>(), y0, 0.0, ts,
                                    tau, nullptr, a),
               std::domain_error);
  stan::math::recover_memory();
}

TEST(user_jacobian_check, pullback) {
  EXPECT_NO_THROW(solve_adjoint_pull<true>());
  stan::math::recover_memory();
  EXPECT_THROW(solve_adjoint_pull<false>(), std::domain_error);
  stan::math::recover_memory();
}
