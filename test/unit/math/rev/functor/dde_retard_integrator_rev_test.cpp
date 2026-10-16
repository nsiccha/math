#include <stan/math/rev.hpp>
#include <test/unit/math/prim/functor/dde_test_functors.hpp>
#include <test/unit/math/rev/util.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

namespace {

using stan::math::coupled_dde_system;
using stan::math::var;
using stan::math::internal::dde_retard_integrator;
using stan::test::dde::dosing_rhs;
using stan::test::dde::dosing_schedule;
using stan::test::dde::run_dosing_schedule;
using stan::test::dde::scalar_lag_rhs;
using stan::test::dde::unit_jump_lag_exact;

using vector_v = Eigen::Matrix<var, Eigen::Dynamic, 1>;

Eigen::VectorXd vec(std::initializer_list<double> values) {
  Eigen::VectorXd v(values.size());
  int i = 0;
  for (double x : values) {
    v(i++) = x;
  }
  return v;
}

// Parameters of the dosing tests: theta = (k, k2, dose_time, dose,
// infusion_duration), then the delays tau_1, tau_2.
struct dosing_parameters {
  std::vector<double> values{0.5, 0.9, 0.4, 2.0, 0.8, 1.3, 2.1};

  std::vector<double> theta() const {
    return {values.begin(), values.begin() + 5};
  }
  std::vector<double> tau() const { return {values[5], values[6]}; }
  dosing_schedule schedule() const {
    dosing_schedule s;
    s.dose_time = values[2];
    s.dose = values[3];
    s.infusion_duration = values[4];
    return s;
  }
};

// The dosing schedule without sensitivities.
std::vector<Eigen::VectorXd> dosing_values(const dosing_parameters& p,
                                           const std::vector<double>& ts) {
  double rate = 0;
  dosing_rhs f{&rate};
  const Eigen::VectorXd y0 = vec({0.0, 0.0});
  const std::vector<double> theta = p.theta();
  const std::vector<double> tau = p.tau();
  const Eigen::VectorXd tau_vec = vec({tau[0], tau[1]});
  coupled_dde_system<dosing_rhs, double, Eigen::VectorXd, std::vector<double>>
      system(f, y0, tau_vec, nullptr, theta);
  return run_dosing_schedule(system, rate, tau, p.schedule(), ts, 1e-12);
}

}  // namespace

// y' = -a y(t - tau) with history 0 and a unit jump at t = 0: on [tau, 2 tau]
// y = 1 - a (t - tau), so dy / dtau = a there. The sensitivity equation
// alone gives 0; the jump condition at t = tau supplies it.
TEST_F(AgradRev, dde_retard_integrator_unit_jump_sensitivities) {
  const double a_val = 0.7;
  const double tau_val = 0.8;
  scalar_lag_rhs f;
  const Eigen::VectorXd y0 = vec({0.0});
  vector_v tau(1);
  tau << tau_val;
  var a = a_val;
  using system_t = coupled_dde_system<scalar_lag_rhs, double, vector_v, var>;
  system_t system(f, y0, tau, nullptr, a);
  ASSERT_EQ(system.size(), 3);
  dde_retard_integrator<system_t> it("unit_jump", system, {tau_val}, 1e-10,
                                     1e-10, 100000);
  const Eigen::VectorXd zero = Eigen::VectorXd::Zero(3);
  it.start(0.0, zero, zero);
  it.jump(vec({1.0, 0.0, 0.0}), Eigen::VectorXd());
  const std::vector<double> ts{0.3, 0.8, 1.0, 1.2, 1.6, 2.0, 2.4, 3.0, 4.5};
  std::vector<Eigen::VectorXd> out;
  it.advance_to(ts.back(), ts,
                [&](size_t, const Eigen::VectorXd& z) { out.push_back(z); });
  ASSERT_EQ(out.size(), ts.size());
  for (size_t i = 0; i < ts.size(); ++i) {
    double y, dy_da, dy_dtau;
    unit_jump_lag_exact(ts[i], a_val, tau_val, y, dy_da, dy_dtau);
    EXPECT_NEAR(out[i](0), y, 1e-9) << ts[i];
    EXPECT_NEAR(out[i](1), dy_da, 1e-9) << ts[i];
    EXPECT_NEAR(out[i](2), dy_dtau, 1e-9) << ts[i];
  }
  // at t = tau the state is the right limit
  EXPECT_NEAR(out[1](2), a_val, 1e-9);
}

// Sensitivities of the dosing schedule with respect to k, k2, the time and
// size of the first bolus, the duration of the infusion (whose end time
// depends on it) and the two delays, against central finite differences.
TEST_F(AgradRev, dde_retard_integrator_dosing_sensitivities) {
  const std::vector<double> ts{0.2, 0.9, 1.6, 2.0, 2.6, 3.4,
                               4.0, 4.6, 5.5, 7.0, 9.0};
  const dosing_parameters p;
  const size_t P = p.values.size();

  double rate = 0;
  dosing_rhs f{&rate};
  const Eigen::VectorXd y0 = vec({0.0, 0.0});
  const std::vector<double> theta_d = p.theta();
  const std::vector<var> theta(theta_d.begin(), theta_d.end());
  const std::vector<double> tau_d = p.tau();
  vector_v tau(2);
  tau << tau_d[0], tau_d[1];
  using system_t
      = coupled_dde_system<dosing_rhs, double, vector_v, std::vector<var>>;
  system_t system(f, y0, tau, nullptr, theta);
  ASSERT_EQ(system.size(), 2 * (1 + P));
  const auto out
      = run_dosing_schedule(system, rate, tau_d, p.schedule(), ts, 1e-12);
  const auto values = dosing_values(p, ts);
  ASSERT_EQ(out.size(), ts.size());
  ASSERT_EQ(values.size(), ts.size());

  for (size_t q = 0; q < P; ++q) {
    const double eps = 1e-6;
    dosing_parameters plus = p;
    dosing_parameters minus = p;
    plus.values[q] += eps;
    minus.values[q] -= eps;
    const auto y_plus = dosing_values(plus, ts);
    const auto y_minus = dosing_values(minus, ts);
    for (size_t i = 0; i < ts.size(); ++i) {
      for (int n = 0; n < 2; ++n) {
        const double fd = (y_plus[i](n) - y_minus[i](n)) / (2 * eps);
        EXPECT_NEAR(out[i](2 + 2 * q + n), fd, 1e-5)
            << "parameter " << q << ", t = " << ts[i] << ", state " << n;
      }
    }
  }
  for (size_t i = 0; i < ts.size(); ++i) {
    EXPECT_NEAR(out[i](0), values[i](0), 1e-12);
    EXPECT_NEAR(out[i](1), values[i](1), 1e-12);
  }
}

// Two jumps at the same time with the same time sensitivities are applied
// one after the other; together they equal one jump of their sum.
TEST_F(AgradRev, dde_retard_integrator_simultaneous_jumps) {
  const dosing_parameters p;
  double rate = 0;
  dosing_rhs f{&rate};
  const Eigen::VectorXd y0 = vec({0.0, 0.0});
  const std::vector<double> theta_d = p.theta();
  const std::vector<var> theta(theta_d.begin(), theta_d.end());
  const std::vector<double> tau_d = p.tau();
  vector_v tau(2);
  tau << tau_d[0], tau_d[1];
  using system_t
      = coupled_dde_system<dosing_rhs, double, vector_v, std::vector<var>>;
  system_t system(f, y0, tau, nullptr, theta);
  const Eigen::Index n = system.size();
  const std::vector<double> ts{1.0, 2.0, 3.0, 4.0, 6.0};
  Eigen::VectorXd du_dtheta = Eigen::VectorXd::Zero(7);
  du_dtheta(2) = 1;

  auto run = [&](const std::vector<double>& parts) {
    dde_retard_integrator<system_t> it("simultaneous", system, tau_d, 1e-10,
                                       1e-10, 100000);
    it.start(0.0, Eigen::VectorXd::Zero(n), Eigen::VectorXd::Zero(n));
    it.advance_to(0.4);
    for (double part : parts) {
      Eigen::VectorXd dz = Eigen::VectorXd::Zero(n);
      dz(0) = part;
      it.jump(dz, du_dtheta);
    }
    std::vector<Eigen::VectorXd> out;
    it.advance_to(ts.back(), ts,
                  [&](size_t, const Eigen::VectorXd& z) { out.push_back(z); });
    return out;
  };
  const auto one = run({2.0});
  const auto two = run({1.2, 0.8});
  ASSERT_EQ(one.size(), two.size());
  for (size_t i = 0; i < one.size(); ++i) {
    for (Eigen::Index k = 0; k < n; ++k) {
      EXPECT_NEAR(one[i](k), two[i](k), 1e-9 * (1 + std::fabs(one[i](k))))
          << ts[i] << " " << k;
    }
  }
}
