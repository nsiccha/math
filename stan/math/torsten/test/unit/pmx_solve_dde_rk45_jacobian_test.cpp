#include <stan/math/rev.hpp>
#include <stan/math/torsten/pmx_solve_dde_rk45.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

/*
 * The pmx_solve_dde_rk45 overload taking the Jacobian of the right hand
 * side, (f, jac, nCmt, ...), against the overload without it, whose
 * sensitivities come from nested autodiff: values and the derivatives with
 * respect to every input that can be a parameter (theta, tau, y_hist,
 * biovar, tlag, time, amt, rate, ii), with two delays, over transient and
 * steady-state schedules.
 *
 * Built once as is and once with -DSTAN_MATH_CHECK_USER_JACOBIAN, which
 * compares every Jacobian evaluation with nested autodiff.
 */

namespace {

using stan::math::var;

/**
 * One-compartment PK with first-order absorption (depot, central) driving an
 * indirect response R whose production is stimulated by the concentration
 * one delay tau_1 earlier and fed back by R one delay tau_2 earlier:
 *
 *   R' = kin (1 + Smax C(t - tau_1) / (SC50 + C(t - tau_1)))
 *            sqrt(kin / kout / R(t - tau_2)) - kout R,
 *
 * theta = (ka, CL, V, kin, kout, Smax, SC50). A nonempty x_r scales CL by
 * x_r[0]; a nonempty x_i with x_i[0] == 0 turns the feedback off. (The
 * model of pmx_solve_dde_rk45_test.cpp.)
 */
struct delayed_indirect_response_rhs {
  template <typename T0, typename T1, typename T2, typename T3>
  Eigen::Matrix<stan::return_type_t<T0, T1, T2, T3>, -1, 1>
  operator()(const T0& t, const Eigen::Matrix<T1, -1, 1>& y,
             const Eigen::Matrix<T2, -1, -1>& y_lag,
             const std::vector<T3>& theta, const std::vector<double>& x_r,
             const std::vector<int>& x_i, std::ostream* msgs) const {
    using scalar = stan::return_type_t<T0, T1, T2, T3>;
    const T3& ka = theta[0];
    const T3& V = theta[2];
    const T3& kin = theta[3];
    const T3& kout = theta[4];
    const T3& Smax = theta[5];
    const T3& SC50 = theta[6];
    const scalar CL = x_r.empty() ? scalar(theta[1]) : theta[1] * x_r[0];
    const scalar C_lag = y_lag(1, 0) / V;
    const scalar feedback = !x_i.empty() && x_i[0] == 0
                                ? scalar(1.0)
                                : stan::math::sqrt(kin / kout / y_lag(2, 1));
    Eigen::Matrix<scalar, -1, 1> dydt(3);
    dydt(0) = -ka * y(0);
    dydt(1) = ka * y(0) - CL / V * y(1);
    dydt(2) = kin * (1 + Smax * C_lag / (SC50 + C_lag)) * feedback - kout * y(2);
    return dydt;
  }
};

/**
 * Jacobian of <code>delayed_indirect_response_rhs</code>,
 * 3 x (3 + 3 * 2 + 7 + size(x_r)): y, y_lag(:, 1), y_lag(:, 2), theta, x_r.
 */
struct delayed_indirect_response_jac {
  Eigen::MatrixXd operator()(double t, const Eigen::VectorXd& y,
                             const Eigen::MatrixXd& y_lag,
                             const std::vector<double>& theta,
                             const std::vector<double>& x_r,
                             const std::vector<int>& x_i,
                             std::ostream* msgs) const {
    const double ka = theta[0], CLp = theta[1], V = theta[2], kin = theta[3],
                 kout = theta[4], Smax = theta[5], SC50 = theta[6];
    const double scale = x_r.empty() ? 1.0 : x_r[0];
    const double CL = CLp * scale;
    const double C_lag = y_lag(1, 0) / V;
    const bool feedback_on = x_i.empty() || x_i[0] != 0;
    const double R_lag = y_lag(2, 1);
    const double fb = feedback_on ? std::sqrt(kin / kout / R_lag) : 1.0;
    const double D = SC50 + C_lag;
    const double S = 1 + Smax * C_lag / D;
    const double dS_dC = Smax * SC50 / (D * D);
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, 16 + x_r.size());
    // y
    J(0, 0) = -ka;
    J(1, 0) = ka;
    J(1, 1) = -CL / V;
    J(2, 2) = -kout;
    // y_lag(:, 1), from column 3; y_lag(:, 2), from column 6
    J(2, 3 + 1) = kin * dS_dC / V * fb;
    if (feedback_on) {
      J(2, 6 + 2) = -kin * S * fb / (2 * R_lag);
    }
    // theta, from column 9
    J(0, 9) = -y(0);
    J(1, 9) = y(0);
    J(1, 10) = -scale / V * y(1);
    J(1, 11) = CL / (V * V) * y(1);
    J(2, 11) = -kin * dS_dC * C_lag / V * fb;
    J(2, 12) = feedback_on ? 1.5 * S * fb : S;
    J(2, 13) = (feedback_on ? -kin * S * fb / (2 * kout) : 0.0) - y(2);
    J(2, 14) = kin * fb * C_lag / D;
    J(2, 15) = -kin * fb * Smax * C_lag / (D * D);
    // x_r: never read by the solver
    if (!x_r.empty()) {
      J(1, 16) = -CLp / V * y(1);
    }
    return J;
  }
};

/**
 * A Jacobian that counts its evaluations, scaling the theta column
 * <code>column</code> (if nonnegative) by <code>factor</code>.
 */
struct counting_jac {
  int* count;
  int column = -1;
  double factor = 1.0;
  Eigen::MatrixXd operator()(double t, const Eigen::VectorXd& y,
                             const Eigen::MatrixXd& y_lag,
                             const std::vector<double>& theta,
                             const std::vector<double>& x_r,
                             const std::vector<int>& x_i,
                             std::ostream* msgs) const {
    ++*count;
    Eigen::MatrixXd J
        = delayed_indirect_response_jac()(t, y, y_lag, theta, x_r, x_i, msgs);
    if (column >= 0) {
      J.col(9 + column) *= factor;
    }
    return J;
  }
};

/**
 * A Jacobian without the columns of the second delay.
 */
struct short_jac {
  Eigen::MatrixXd operator()(double t, const Eigen::VectorXd& y,
                             const Eigen::MatrixXd& y_lag,
                             const std::vector<double>& theta,
                             const std::vector<double>& x_r,
                             const std::vector<int>& x_i,
                             std::ostream* msgs) const {
    const Eigen::MatrixXd J
        = delayed_indirect_response_jac()(t, y, y_lag, theta, x_r, x_i, msgs);
    Eigen::MatrixXd res(3, J.cols() - 3);
    res << J.leftCols(6), J.rightCols(J.cols() - 9);
    return res;
  }
};

/**
 * A NONMEM-style event schedule with its parameters.
 */
struct problem {
  std::vector<double> time, amt, rate, ii;
  std::vector<int> evid, cmt, addl, ss;
  std::vector<std::vector<double>> theta, biovar, tlag, x_r;
  std::vector<std::vector<int>> x_i;
  std::vector<double> tau{2.0, 5.0};
  std::vector<double> y_hist{0.0, 0.0, 7.0};

  void add(double t, int ev, int c, double a = 0, double r = 0, double i = 0,
           int ad = 0, int s = 0) {
    time.push_back(t);
    evid.push_back(ev);
    cmt.push_back(c);
    amt.push_back(a);
    rate.push_back(r);
    ii.push_back(i);
    addl.push_back(ad);
    ss.push_back(s);
  }

  void observe(const std::vector<double>& ts) {
    for (double t : ts) {
      add(t, 0, 2);
    }
  }
};

const std::vector<double> theta_ref{0.9, 4.0, 30.0, 2.0, 0.25, 1.5, 2.0};

/**
 * Oral doses into the depot every 24 h, an IV bolus into the central
 * compartment at 30 and an infusion into it at 55, with a theta row that
 * changes at the bolus, CL scaled by x_r, lag times and bioavailability.
 */
problem transient_problem() {
  problem p;
  p.add(0, 1, 1, 100, 0, 24, 3);
  p.observe({1.5, 3.3, 7.1, 11.7, 18.2, 26.9});
  p.add(30, 1, 2, 40);
  p.observe({31.4, 36.6, 44.2, 50.8});
  p.add(55, 1, 2, 60, 20);
  p.observe({57.3, 63.9, 75.5, 88.1, 99.4});
  const size_t n = p.time.size();
  p.theta.assign(n, theta_ref);
  for (size_t i = 7; i < n; ++i) {
    p.theta[i][1] = 5.5;  // CL changes at the bolus
  }
  p.biovar.assign(1, {0.8, 0.95, 1.0});
  p.tlag.assign(1, {0.7, 0.4, 0.0});
  p.x_r.assign(n, {1.1});
  p.x_i.assign(n, {1});
  return p;
}

/**
 * Steady states: periodic oral doses (SS = 1, II = 24), a periodic
 * infusion (SS = 1, II = 12) and a constant infusion (SS = 1, II = 0).
 */
problem steady_state_problem() {
  problem p;
  p.add(0, 1, 1, 100, 0, 24, 2, 1);
  p.observe({3.3, 11.7, 30.2, 55.1});
  p.add(60, 1, 2, 60, 20, 12, 1, 1);
  p.observe({61.4, 66.2, 75.3, 80.0});
  p.add(90, 1, 2, 0, 5, 0, 0, 1);
  p.observe({91.7, 99.9});
  const size_t n = p.time.size();
  p.theta.assign(n, theta_ref);
  p.biovar.assign(1, {0.8, 0.95, 1.0});
  p.tlag.assign(1, {0.7, 0.0, 0.0});
  p.x_r.assign(n, {1.1});
  p.x_i.assign(n, {1});
  return p;
}

/**
 * Values (column-major) and the derivatives of every value with respect to
 * every parameter input (theta, tau, y_hist, biovar, tlag, time, amt, rate,
 * ii, in that order), with solve(...) called on autodiff copies of all of
 * them.
 */
template <typename Solve>
void values_and_jacobian(const Solve& solve, const problem& p,
                         Eigen::VectorXd& values, Eigen::MatrixXd& jac) {
  stan::math::nested_rev_autodiff nested;
  std::vector<var> inputs;
  auto vars_1d = [&](const std::vector<double>& x) {
    std::vector<var> v(x.begin(), x.end());
    inputs.insert(inputs.end(), v.begin(), v.end());
    return v;
  };
  auto vars_2d = [&](const std::vector<std::vector<double>>& x) {
    std::vector<std::vector<var>> v;
    for (const auto& row : x) {
      v.push_back(vars_1d(row));
    }
    return v;
  };
  auto vars_eigen = [&](const std::vector<double>& x) {
    const std::vector<var> v = vars_1d(x);
    return Eigen::Matrix<var, -1, 1>(Eigen::Map<const Eigen::Matrix<var, -1, 1>>(v.data(), v.size()));
  };
  const auto theta = vars_2d(p.theta);
  const auto tau = vars_eigen(p.tau);
  const auto y_hist = vars_eigen(p.y_hist);
  const auto biovar = vars_2d(p.biovar);
  const auto tlag = vars_2d(p.tlag);
  const auto time = vars_1d(p.time);
  const auto amt = vars_1d(p.amt);
  const auto rate = vars_1d(p.rate);
  const auto ii = vars_1d(p.ii);
  Eigen::Matrix<var, -1, -1> y
      = solve(theta, tau, y_hist, biovar, tlag, time, amt, rate, ii);
  values.resize(y.size());
  jac.resize(y.size(), inputs.size());
  for (Eigen::Index i = 0; i < y.size(); ++i) {
    nested.set_zero_all_adjoints();
    values(i) = y(i).val();
    y(i).grad();
    for (size_t k = 0; k < inputs.size(); ++k) {
      jac(i, k) = inputs[k].adj();
    }
  }
}

/**
 * Expect the jac overload's values and derivatives within tol of the
 * autodiff overload's, relative to 1 + |autodiff value|, and the
 * comparison to be meaningful: nonzero derivatives with respect to most
 * inputs.
 */
template <typename SolveAD, typename SolveJac>
void expect_jacobian_path_matches(const SolveAD& solve_ad, const SolveJac& solve_jac,
                                  const problem& p, double tol,
                                  const std::string& name,
                                  double min_nonzero_fraction = 0.5) {
  Eigen::VectorXd values_ad, values_jac;
  Eigen::MatrixXd jac_ad, jac_jac;
  values_and_jacobian(solve_ad, p, values_ad, jac_ad);
  values_and_jacobian(solve_jac, p, values_jac, jac_jac);
  ASSERT_EQ(values_ad.size(), values_jac.size());
  ASSERT_EQ(jac_ad.cols(), jac_jac.cols());
  for (Eigen::Index i = 0; i < values_ad.size(); ++i) {
    EXPECT_NEAR(values_jac(i), values_ad(i), tol * (1 + std::abs(values_ad(i))))
        << name << " value " << i;
  }
  double max_diff = 0;
  for (Eigen::Index i = 0; i < jac_ad.rows(); ++i) {
    for (Eigen::Index k = 0; k < jac_ad.cols(); ++k) {
      const double scale = 1 + std::abs(jac_ad(i, k));
      max_diff = std::max(max_diff, std::abs(jac_jac(i, k) - jac_ad(i, k)) / scale);
      EXPECT_NEAR(jac_jac(i, k), jac_ad(i, k), tol * scale)
          << name << " d value " << i << " / d input " << k;
    }
  }
  // positive control: the comparison covers derivatives that are not zero
  int nonzero_inputs = 0;
  for (Eigen::Index k = 0; k < jac_ad.cols(); ++k) {
    nonzero_inputs += jac_ad.col(k).cwiseAbs().maxCoeff() > 1e-8;
  }
  EXPECT_GT(nonzero_inputs, min_nonzero_fraction * jac_ad.cols()) << name;
  std::cout << name << ": " << jac_ad.rows() << " values x " << jac_ad.cols()
            << " inputs (" << nonzero_inputs << " with nonzero derivatives), "
            << "largest relative difference " << max_diff << "\n";
}

const double dde_rtol = 1e-11;
const int max_num_steps = 10000000;
const double ss_rtol = 1e-12;
const int ss_max_num_steps = 1000;

/**
 * The solve of one overload: with or without jac, x_r and x_i passed.
 */
template <typename... Jac>
auto full_solve(const problem& p, const Jac&... jac) {
  return [&p, jac...](const auto& theta, const auto& tau, const auto& y_hist,
                      const auto& biovar, const auto& tlag, const auto& time,
                      const auto& amt, const auto& rate, const auto& ii) {
    return torsten::pmx_solve_dde_rk45(
        delayed_indirect_response_rhs(), jac..., 3, time, amt, rate, ii, p.evid,
        p.cmt, p.addl, p.ss, theta, tau, y_hist, biovar, tlag, p.x_r, p.x_i,
        dde_rtol, dde_rtol, max_num_steps, ss_rtol, ss_rtol, ss_max_num_steps,
        nullptr);
  };
}

template <typename T>
Eigen::Matrix<T, -1, 1> to_eigen(const std::vector<T>& x) {
  return Eigen::Map<const Eigen::Matrix<T, -1, 1>>(x.data(), x.size());
}

}  // namespace

TEST(pmx_solve_dde_rk45_jacobian, transient_schedule) {
  const problem p = transient_problem();
  int count = 0;
  expect_jacobian_path_matches(full_solve(p), full_solve(p, counting_jac{&count}),
                               p, 1e-6, "pmx_solve_dde_rk45 transient");
  EXPECT_GT(count, 0) << "the Jacobian overload never evaluated jac";
}

TEST(pmx_solve_dde_rk45_jacobian, transient_schedule_feedback_off) {
  problem p = transient_problem();
  p.x_r.assign(1, {});
  p.x_i.assign(1, {0});
  int count = 0;
  expect_jacobian_path_matches(full_solve(p), full_solve(p, counting_jac{&count}),
                               p, 1e-6, "pmx_solve_dde_rk45 transient, no feedback");
  EXPECT_GT(count, 0) << "the Jacobian overload never evaluated jac";
}

TEST(pmx_solve_dde_rk45_jacobian, steady_state_schedule) {
  const problem p = steady_state_problem();
  int count = 0;
  expect_jacobian_path_matches(full_solve(p), full_solve(p, counting_jac{&count}),
                               p, 1e-6, "pmx_solve_dde_rk45 steady state");
  EXPECT_GT(count, 0) << "the Jacobian overload never evaluated jac";
}

/*
 * The shorter overloads: without x_i, without x_r, without tlag, without
 * biovar, with only the stream as control.
 */
TEST(pmx_solve_dde_rk45_jacobian, overloads) {
  const problem p = transient_problem();
  std::vector<std::vector<var>> theta;
  for (const auto& row : p.theta) {
    theta.emplace_back(row.begin(), row.end());
  }
  const Eigen::VectorXd tau = to_eigen(p.tau);
  const Eigen::VectorXd y_hist = to_eigen(p.y_hist);
  auto expect_same = [](const Eigen::Matrix<var, -1, -1>& a,
                        const Eigen::Matrix<var, -1, -1>& b, const char* what) {
    ASSERT_EQ(a.size(), b.size());
    for (Eigen::Index i = 0; i < a.size(); ++i) {
      EXPECT_NEAR(a(i).val(), b(i).val(), 1e-6 * (1 + std::abs(a(i).val()))) << what;
    }
  };
  const delayed_indirect_response_rhs f;
  const delayed_indirect_response_jac jac;
  auto solve = [&](const auto&... args) {
    return torsten::pmx_solve_dde_rk45(args...);
  };
  expect_same(solve(f, 3, p.time, p.amt, p.rate, p.ii, p.evid, p.cmt, p.addl, p.ss,
                    theta, tau, y_hist, p.biovar, p.tlag, p.x_r, dde_rtol,
                    dde_rtol, max_num_steps, nullptr),
              solve(f, jac, 3, p.time, p.amt, p.rate, p.ii, p.evid, p.cmt, p.addl,
                    p.ss, theta, tau, y_hist, p.biovar, p.tlag, p.x_r, dde_rtol,
                    dde_rtol, max_num_steps, nullptr),
              "without x_i");
  expect_same(solve(f, 3, p.time, p.amt, p.rate, p.ii, p.evid, p.cmt, p.addl, p.ss,
                    theta, tau, y_hist, p.biovar, p.tlag, nullptr),
              solve(f, jac, 3, p.time, p.amt, p.rate, p.ii, p.evid, p.cmt, p.addl,
                    p.ss, theta, tau, y_hist, p.biovar, p.tlag, nullptr),
              "without x_r, default tolerances");
  expect_same(solve(f, 3, p.time, p.amt, p.rate, p.ii, p.evid, p.cmt, p.addl, p.ss,
                    theta[0], tau, y_hist, p.biovar[0], nullptr),
              solve(f, jac, 3, p.time, p.amt, p.rate, p.ii, p.evid, p.cmt, p.addl,
                    p.ss, theta[0], tau, y_hist, p.biovar[0], nullptr),
              "1-d theta and biovar, without tlag");
  expect_same(solve(f, 3, p.time, p.amt, p.rate, p.ii, p.evid, p.cmt, p.addl, p.ss,
                    theta, tau, y_hist, nullptr),
              solve(f, jac, 3, p.time, p.amt, p.rate, p.ii, p.evid, p.cmt, p.addl,
                    p.ss, theta, tau, y_hist, nullptr),
              "without biovar");
}

/*
 * The Jacobian drives the derivatives: scaling its dkin column changes the
 * derivatives with respect to kin; with STAN_MATH_CHECK_USER_JACOBIAN the
 * scaled Jacobian is refused. A Jacobian of the wrong size is refused in
 * either build.
 */
TEST(pmx_solve_dde_rk45_jacobian, wrong_jacobian) {
  const problem p = transient_problem();
  int count = 0;
  const counting_jac scaled{&count, 3, 1.5};
  auto solve_d_kin = [&](const auto& jac) {
    stan::math::nested_rev_autodiff nested;
    std::vector<var> theta_v(theta_ref.begin(), theta_ref.end());
    auto y = torsten::pmx_solve_dde_rk45(
        delayed_indirect_response_rhs(), jac, 3, p.time, p.amt, p.rate, p.ii,
        p.evid, p.cmt, p.addl, p.ss, theta_v, to_eigen(p.tau),
        to_eigen(p.y_hist), p.biovar, p.tlag, p.x_r, p.x_i, dde_rtol, dde_rtol,
        max_num_steps, nullptr);
    y(2, 10).grad();
    return theta_v[3].adj();
  };
#ifdef STAN_MATH_CHECK_USER_JACOBIAN
  EXPECT_THROW(solve_d_kin(scaled), std::domain_error);
#else
  const double right = solve_d_kin(counting_jac{&count});
  const double wrong = solve_d_kin(scaled);
  EXPECT_GT(std::abs(wrong - right), 1e-3 * std::abs(right));
#endif
  EXPECT_THROW(solve_d_kin(short_jac()), std::invalid_argument);
}

/*
 * theta as data, the other inputs as autodiff variables: no theta column
 * of the Jacobian is read.
 */
TEST(pmx_solve_dde_rk45_jacobian, data_theta) {
  const problem p = transient_problem();
  auto data_theta = [&p](const auto& solve) {
    return [&p, solve](const auto& theta, const auto&... rest) {
      return solve(p.theta, rest...);
    };
  };
  int count = 0;
  expect_jacobian_path_matches(data_theta(full_solve(p)),
                               data_theta(full_solve(p, counting_jac{&count})),
                               p, 1e-6, "pmx_solve_dde_rk45 data theta", 0.1);
  EXPECT_GT(count, 0) << "the Jacobian overload never evaluated jac";
}
