#include <stan/math/prim.hpp>
#include <test/unit/math/prim/functor/dde_test_functors.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace {

using stan::math::coupled_dde_system;
using stan::math::internal::dde_breakpoints;
using stan::math::internal::dde_retard_integrator;
using stan::test::dde::dosing_rhs;
using stan::test::dde::dosing_schedule;
using stan::test::dde::run_dosing_schedule;
using stan::test::dde::scalar_lag_rhs;
using stan::test::dde::unit_jump_lag_exact;

using scalar_system
    = coupled_dde_system<scalar_lag_rhs, double, Eigen::VectorXd, double>;
using dosing_system = coupled_dde_system<dosing_rhs, double, Eigen::VectorXd,
                                         std::vector<double>>;

Eigen::VectorXd vec(std::initializer_list<double> values) {
  Eigen::VectorXd v(values.size());
  int i = 0;
  for (double x : values) {
    v(i++) = x;
  }
  return v;
}

std::vector<Eigen::VectorXd> collect(dde_retard_integrator<scalar_system>& it,
                                     double t1, const std::vector<double>& ts) {
  std::vector<Eigen::VectorXd> out;
  it.advance_to(t1, ts,
                [&](size_t, const Eigen::VectorXd& z) { out.push_back(z); });
  return out;
}

}  // namespace

// The breakpoints of a source u are u + sum_j k_j tau_j, 1 <= sum_j k_j <= 5,
// up to the end time; there are none of higher order (the last one is
// 0.5 + 5 * 2.5).
TEST(dde_retard_integrator_prim, breakpoints_of_a_source) {
  const std::vector<double> tau{1.0, 2.5};
  dde_breakpoints breakpoints(tau);
  breakpoints.add_source(0.5);
  const double tol = 1e-12;
  std::vector<double> got;
  double x = 0;
  while (x < 20) {
    x = breakpoints.next(x, 20, tol);
    got.push_back(x);
  }
  const std::vector<double> expected{1.5, 2.5,  3.0,  3.5,  4.0,  4.5, 5.0,
                                     5.5, 6.0,  6.5,  7.0,  7.5,  8.0, 8.5,
                                     9.0, 10.0, 10.5, 11.5, 13.0, 20.0};
  ASSERT_EQ(got.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(got[i], expected[i], 1e-14) << i;
  }
}

// y' = -a y(t - tau) with history 0 and a unit jump at t = 0, against the
// method of steps.
TEST(dde_retard_integrator_prim, unit_jump_matches_exact_solution) {
  const double a = 0.7;
  const double tau = 0.8;
  scalar_lag_rhs f;
  const Eigen::VectorXd y0 = vec({0.0});
  const Eigen::VectorXd tau_vec = vec({tau});
  scalar_system system(f, y0, tau_vec, nullptr, a);
  dde_retard_integrator<scalar_system> it("unit_jump", system, {tau}, 1e-10,
                                          1e-10, 100000);
  it.start(0.0, y0, y0);
  it.jump(vec({1.0}), Eigen::VectorXd());
  const std::vector<double> ts{0.3, 0.8, 1.0, 1.2, 1.6, 2.0, 2.4, 3.0, 4.5};
  const auto out = collect(it, ts.back(), ts);
  ASSERT_EQ(out.size(), ts.size());
  for (size_t i = 0; i < ts.size(); ++i) {
    double y, dy_da, dy_dtau;
    unit_jump_lag_exact(ts[i], a, tau, y, dy_da, dy_dtau);
    EXPECT_NEAR(out[i](0), y, 1e-9) << ts[i];
  }
  EXPECT_EQ(it.time(), ts.back());
  EXPECT_EQ(it.state()(0), out.back()(0));
}

// A history different from the initial state is the same as a jump at t0.
TEST(dde_retard_integrator_prim, initial_history_is_a_jump_at_t0) {
  scalar_lag_rhs f;
  const Eigen::VectorXd zero = vec({0.0});
  const Eigen::VectorXd tau_vec = vec({0.8});
  scalar_system system(f, zero, tau_vec, nullptr, 0.7);
  const std::vector<double> ts{0.5, 1.0, 2.0, 3.5};

  dde_retard_integrator<scalar_system> by_history("by_history", system, {0.8},
                                                  1e-8, 1e-8, 100000);
  by_history.start(0.0, vec({1.0}), zero);
  const auto a = collect(by_history, ts.back(), ts);

  dde_retard_integrator<scalar_system> by_jump("by_jump", system, {0.8}, 1e-8,
                                               1e-8, 100000);
  by_jump.start(0.0, zero, zero);
  by_jump.jump(vec({1.0}), Eigen::VectorXd());
  const auto b = collect(by_jump, ts.back(), ts);

  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i](0), b[i](0));
  }
}

// The dosing schedule (a bolus, an infusion and a second bolus, two delays)
// against DelayDiffEq.jl, MethodOfSteps(Vern9()) with reltol = abstol =
// 1e-13, the doses as callbacks and the infusion rate switched by callbacks,
// and tstops at the events and their propagation u + k1 tau1 + k2 tau2
// (1 <= k1 + k2 <= 5); it agrees with the solve at 1e-11 to 6e-11.
TEST(dde_retard_integrator_prim, dosing_schedule_matches_delaydiffeq) {
  const std::vector<double> ts{0.2, 0.9, 1.6, 2.0, 2.6, 3.4,
                               4.0, 4.6, 5.5, 7.0, 9.0};
  const std::vector<std::vector<double>> reference{
      {0.0, 0.0},
      {1.557601566142812, 0.0},
      {1.2927055741851985, 0.0},
      {1.783454795948826, 0.17062075492817208},
      {1.8007748315744438, 0.4305931039337163},
      {2.025826221078808, 0.8121965134435846},
      {1.500768976509955, 0.968805212135372},
      {1.111797002832432, 1.1396590838264529},
      {0.7089130678946689, 1.2552655295407777},
      {0.33486682207446233, 1.0300244203378106},
      {0.12319061937161065, 0.5526211765881972}};
  double rate = 0;
  dosing_rhs f{&rate};
  const Eigen::VectorXd y0 = vec({0.0, 0.0});
  const std::vector<double> tau{1.3, 2.1};
  const Eigen::VectorXd tau_vec = vec({1.3, 2.1});
  const std::vector<double> theta{0.5, 0.9};
  dosing_system system(f, y0, tau_vec, nullptr, theta);
  const auto out
      = run_dosing_schedule(system, rate, tau, dosing_schedule(), ts, 1e-10);
  ASSERT_EQ(out.size(), ts.size());
  for (size_t i = 0; i < ts.size(); ++i) {
    for (int n = 0; n < 2; ++n) {
      EXPECT_NEAR(out[i](n), reference[i][n], 1e-8) << ts[i] << " " << n;
    }
  }
}

// Stopping at intermediate times only changes the result within the
// tolerance.
TEST(dde_retard_integrator_prim, advance_in_parts_matches_one_call) {
  scalar_lag_rhs f;
  const Eigen::VectorXd y0 = vec({1.0});
  const Eigen::VectorXd tau_vec = vec({0.6});
  scalar_system system(f, y0, tau_vec, nullptr, 1.1);
  const std::vector<double> ts{0.5, 1.7, 2.9, 4.1, 6.0};

  dde_retard_integrator<scalar_system> whole("whole", system, {0.6}, 1e-12,
                                             1e-12, 100000);
  whole.start(0.0, y0, y0);
  const auto a = collect(whole, ts.back(), ts);

  dde_retard_integrator<scalar_system> parts("parts", system, {0.6}, 1e-12,
                                             1e-12, 100000);
  parts.start(0.0, y0, y0);
  std::vector<Eigen::VectorXd> b = collect(parts, 1.0, {0.5});
  parts.advance_to(1.33);
  for (const auto& z : collect(parts, 3.5, {1.7, 2.9})) {
    b.push_back(z);
  }
  for (const auto& z : collect(parts, 6.0, {4.1, 6.0})) {
    b.push_back(z);
  }
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(a[i](0), b[i](0), 1e-10) << ts[i];
  }
}

// start() again forgets the history: the result is that of a new integrator.
TEST(dde_retard_integrator_prim, restart_forgets_the_history) {
  scalar_lag_rhs f;
  const Eigen::VectorXd y0 = vec({1.0});
  const Eigen::VectorXd tau_vec = vec({0.6});
  scalar_system system(f, y0, tau_vec, nullptr, 1.1);
  const std::vector<double> ts{2.5, 3.0, 4.2};

  dde_retard_integrator<scalar_system> restarted("restarted", system, {0.6},
                                                 1e-8, 1e-8, 100000);
  restarted.start(0.0, y0, y0);
  restarted.advance_to(2.0);
  const Eigen::VectorXd z = restarted.state();
  restarted.start(2.0, z, vec({0.25}));
  const auto a = collect(restarted, ts.back(), ts);

  dde_retard_integrator<scalar_system> fresh("fresh", system, {0.6}, 1e-8, 1e-8,
                                             100000);
  fresh.start(2.0, z, vec({0.25}));
  const auto b = collect(fresh, ts.back(), ts);

  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i](0), b[i](0));
  }
}

TEST(dde_retard_integrator_prim, errors) {
  scalar_lag_rhs f;
  const Eigen::VectorXd y0 = vec({1.0});
  const Eigen::VectorXd tau_vec = vec({0.6});
  scalar_system system(f, y0, tau_vec, nullptr, 1.1);
  dde_retard_integrator<scalar_system> it("errors", system, {0.6}, 1e-8, 1e-8,
                                          100000);
  EXPECT_THROW_MSG(it.advance_to(1.0), std::domain_error,
                   "advance_to called before start");
  EXPECT_THROW_MSG(it.start(0.0, vec({1.0, 2.0}), y0), std::invalid_argument,
                   "state");
  EXPECT_THROW_MSG(it.start(0.0, y0, vec({1.0, 2.0})), std::invalid_argument,
                   "history");
  it.start(0.0, y0, y0);
  it.advance_to(1.0);
  it.advance_to(1.0);
  EXPECT_EQ(it.time(), 1.0);
  EXPECT_THROW_MSG(it.advance_to(0.5), std::domain_error, "end time");
  EXPECT_THROW_MSG(it.jump(vec({1.0, 2.0}), Eigen::VectorXd()),
                   std::invalid_argument, "state jump");
  EXPECT_THROW_MSG(it.jump(vec({1.0}), vec({1.0})), std::invalid_argument,
                   "time sensitivities");
}
