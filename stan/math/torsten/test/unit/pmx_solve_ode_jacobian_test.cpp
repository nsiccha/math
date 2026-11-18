#include <stan/math/rev.hpp>
#include <stan/math/torsten/pmx_solve_rk45.hpp>
#include <stan/math/torsten/pmx_solve_bdf.hpp>
#include <stan/math/torsten/pmx_solve_adams.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

/*
 * The pmx_solve_rk45/bdf/adams overloads taking the Jacobian of the right
 * hand side, (f, jac, nCmt, ...), against the overloads without it, whose
 * sensitivities come from nested autodiff: values and the derivatives with
 * respect to every input that can be a parameter (theta, biovar, tlag,
 * time, amt, rate, ii) over transient and steady-state schedules.
 *
 * Built once as is and once with -DSTAN_MATH_CHECK_USER_JACOBIAN, which
 * compares every Jacobian evaluation with nested autodiff.
 */

namespace {

using stan::math::var;

/**
 * Two-compartment PK with first-order absorption (gut, central,
 * peripheral) and an indirect response R whose production is inhibited by
 * the concentration:
 *
 *   R' = kin IC50 / (IC50 + C) - kout R,  C = central / V1,
 *
 * theta = (CL, Q, V1, V2, ka, kin, kout, IC50). A nonempty x_r scales CL
 * by (x_r[0] / 70)^0.75.
 */
struct twocpt_ir_rhs {
  template <typename T0, typename T1, typename T2>
  Eigen::Matrix<stan::return_type_t<T0, T1, T2>, -1, 1>
  operator()(const T0& t, const Eigen::Matrix<T1, -1, 1>& y,
             const std::vector<T2>& theta, const std::vector<double>& x_r,
             const std::vector<int>& x_i, std::ostream* msgs) const {
    using scalar = stan::return_type_t<T0, T1, T2>;
    const double s = x_r.empty() ? 1.0 : std::pow(x_r[0] / 70.0, 0.75);
    const T2 &CL = theta[0], &Q = theta[1], &V1 = theta[2], &V2 = theta[3],
             &ka = theta[4], &kin = theta[5], &kout = theta[6], &IC50 = theta[7];
    const scalar C = y(1) / V1;
    Eigen::Matrix<scalar, -1, 1> dydt(4);
    dydt(0) = -ka * y(0);
    dydt(1) = ka * y(0) - (s * CL + Q) / V1 * y(1) + Q / V2 * y(2);
    dydt(2) = Q / V1 * y(1) - Q / V2 * y(2);
    dydt(3) = kin * IC50 / (IC50 + C) - kout * y(3);
    return dydt;
  }
};

/**
 * Jacobian of <code>twocpt_ir_rhs</code>: 4 x (4 + 8 + size(x_r)).
 */
struct twocpt_ir_jac {
  Eigen::MatrixXd operator()(double t, const Eigen::VectorXd& y,
                             const std::vector<double>& theta,
                             const std::vector<double>& x_r,
                             const std::vector<int>& x_i,
                             std::ostream* msgs) const {
    const double s = x_r.empty() ? 1.0 : std::pow(x_r[0] / 70.0, 0.75);
    const double CL = theta[0], Q = theta[1], V1 = theta[2], V2 = theta[3],
                 ka = theta[4], kin = theta[5], kout = theta[6], IC50 = theta[7];
    const double C = y(1) / V1;
    const double D = IC50 + C;
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(4, 12 + x_r.size());
    // states
    J(0, 0) = -ka;
    J(1, 0) = ka;
    J(1, 1) = -(s * CL + Q) / V1;
    J(1, 2) = Q / V2;
    J(2, 1) = Q / V1;
    J(2, 2) = -Q / V2;
    J(3, 1) = -kin * IC50 / (D * D) / V1;
    J(3, 3) = -kout;
    // theta, from column 4
    J(0, 8) = -y(0);
    J(1, 4) = -s * y(1) / V1;
    J(1, 5) = -y(1) / V1 + y(2) / V2;
    J(1, 6) = (s * CL + Q) * y(1) / (V1 * V1);
    J(1, 7) = -Q * y(2) / (V2 * V2);
    J(1, 8) = y(0);
    J(2, 5) = y(1) / V1 - y(2) / V2;
    J(2, 6) = -Q * y(1) / (V1 * V1);
    J(2, 7) = Q * y(2) / (V2 * V2);
    J(3, 6) = kin * IC50 * C / (V1 * D * D);
    J(3, 9) = IC50 / D;
    J(3, 10) = -y(3);
    J(3, 11) = kin * C / (D * D);
    // x_r: never read by the solvers
    if (!x_r.empty()) {
      J(1, 12) = -0.75 * s / x_r[0] * CL * y(1) / V1;
    }
    return J;
  }
};

/**
 * The same right hand side with the stream as third argument,
 * f(t, y, msgs, theta, x_r, x_i), and its Jacobian in that order.
 */
struct twocpt_ir_rhs_stream_third {
  template <typename T0, typename T1, typename T2>
  Eigen::Matrix<stan::return_type_t<T0, T1, T2>, -1, 1>
  operator()(const T0& t, const Eigen::Matrix<T1, -1, 1>& y, std::ostream* msgs,
             const std::vector<T2>& theta, const std::vector<double>& x_r,
             const std::vector<int>& x_i) const {
    return twocpt_ir_rhs()(t, y, theta, x_r, x_i, msgs);
  }
};

struct twocpt_ir_jac_stream_third {
  Eigen::MatrixXd operator()(double t, const Eigen::VectorXd& y, std::ostream* msgs,
                             const std::vector<double>& theta,
                             const std::vector<double>& x_r,
                             const std::vector<int>& x_i) const {
    return twocpt_ir_jac()(t, y, theta, x_r, x_i, msgs);
  }
};

/**
 * A Jacobian that counts its evaluations, scaling the theta column
 * <code>column</code> (if nonnegative) by <code>factor</code>, and the
 * state column <code>state_column</code> (if nonnegative) too.
 */
struct counting_jac {
  int* count;
  int column = -1;
  double factor = 1.0;
  int state_column = -1;
  Eigen::MatrixXd operator()(double t, const Eigen::VectorXd& y,
                             const std::vector<double>& theta,
                             const std::vector<double>& x_r,
                             const std::vector<int>& x_i,
                             std::ostream* msgs) const {
    ++*count;
    Eigen::MatrixXd J = twocpt_ir_jac()(t, y, theta, x_r, x_i, msgs);
    if (column >= 0) {
      J.col(4 + column) *= factor;
    }
    if (state_column >= 0) {
      J.col(state_column) *= factor;
    }
    return J;
  }
};

/**
 * A Jacobian with one column too few.
 */
struct short_jac {
  Eigen::MatrixXd operator()(double t, const Eigen::VectorXd& y,
                             const std::vector<double>& theta,
                             const std::vector<double>& x_r,
                             const std::vector<int>& x_i,
                             std::ostream* msgs) const {
    const Eigen::MatrixXd J = twocpt_ir_jac()(t, y, theta, x_r, x_i, msgs);
    return J.leftCols(J.cols() - 1);
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

const std::vector<double> theta_ref{5.0, 8.0, 35.0, 105.0, 1.2, 4.0, 0.4, 0.6};

/**
 * Oral doses with ADDL/II, an IV bolus and an IV infusion, with a
 * covariate (x_r) and a theta row that changes at the bolus; lag time and
 * bioavailability on the gut.
 */
problem transient_problem() {
  problem p;
  p.add(0, 1, 1, 80, 0, 12, 2);
  p.observe({0.5, 1.5, 4.0, 9.0, 13.1, 20.0, 27.5});
  p.add(30, 1, 2, 30);
  p.observe({31.0, 34.0, 40.0});
  p.add(44, 1, 2, 50, 10);
  p.observe({45.0, 47.5, 52.0, 60.0, 72.0});
  const size_t n = p.time.size();
  p.theta.assign(n, theta_ref);
  for (size_t i = 8; i < n; ++i) {
    p.theta[i][0] = 6.5;  // CL changes at the bolus
  }
  p.biovar.assign(1, {0.8, 1.0, 1.0, 1.0});
  p.tlag.assign(1, {0.4, 0.0, 0.0, 0.0});
  p.x_r.assign(n, {82.0});
  p.x_i.assign(n, std::vector<int>{});
  return p;
}

/**
 * Steady states: periodic oral doses (SS = 1, II = 12), a periodic
 * infusion (SS = 1, II = 24), and a constant infusion (SS = 1, II = 0).
 */
problem steady_state_problem() {
  problem p;
  p.add(0, 1, 1, 80, 0, 12, 1, 1);
  p.observe({1.0, 5.0, 11.0, 18.0});
  p.add(30, 1, 2, 60, 15, 24, 1, 1);
  p.observe({31.0, 35.0, 50.0, 60.0});
  p.add(80, 1, 2, 0, 3, 0, 0, 1);
  p.observe({82.0, 90.0});
  const size_t n = p.time.size();
  p.theta.assign(n, theta_ref);
  p.biovar.assign(1, {0.8, 1.0, 1.0, 1.0});
  p.tlag.assign(1, {0.4, 0.0, 0.0, 0.0});
  p.x_r.assign(n, {82.0});
  p.x_i.assign(n, std::vector<int>{});
  return p;
}

/**
 * Values (column-major) and the derivatives of every value with respect to
 * every parameter input (theta, biovar, tlag, time, amt, rate, ii, in that
 * order), with solve(theta, biovar, tlag, time, amt, rate, ii) called on
 * autodiff copies of all of them.
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
  const auto theta = vars_2d(p.theta);
  const auto biovar = vars_2d(p.biovar);
  const auto tlag = vars_2d(p.tlag);
  const auto time = vars_1d(p.time);
  const auto amt = vars_1d(p.amt);
  const auto rate = vars_1d(p.rate);
  const auto ii = vars_1d(p.ii);
  Eigen::Matrix<var, -1, -1> y = solve(theta, biovar, tlag, time, amt, rate, ii);
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

/**
 * The solvers, as functors of (f, jac or none, ...).
 */
struct rk45 {
  static constexpr const char* name = "pmx_solve_rk45";
  template <typename... Args>
  auto operator()(const Args&... args) const {
    return torsten::pmx_solve_rk45(args...);
  }
};

struct bdf {
  static constexpr const char* name = "pmx_solve_bdf";
  template <typename... Args>
  auto operator()(const Args&... args) const {
    return torsten::pmx_solve_bdf(args...);
  }
};

struct adams {
  static constexpr const char* name = "pmx_solve_adams";
  template <typename... Args>
  auto operator()(const Args&... args) const {
    return torsten::pmx_solve_adams(args...);
  }
};

template <typename Solver>
class pmx_solve_ode_jacobian : public ::testing::Test {};

using solvers = ::testing::Types<rk45, bdf, adams>;
TYPED_TEST_SUITE(pmx_solve_ode_jacobian, solvers);

const double ode_rtol = 1e-10;
const double ode_atol = 1e-10;
const int max_num_steps = 100000000;
const double as_rtol = 1e-8;
const double as_atol = 1e-8;
const int as_max_num_steps = 100;

/**
 * The solve of one overload: with or without jac (std::tuple<>), x_r and
 * x_i passed.
 */
template <typename Solver, typename... Jac>
auto full_solve(const problem& p, const Jac&... jac) {
  return [&p, jac...](const auto& theta, const auto& biovar, const auto& tlag,
                      const auto& time, const auto& amt, const auto& rate,
                      const auto& ii) {
    return Solver()(twocpt_ir_rhs(), jac..., 4, time, amt, rate, ii, p.evid, p.cmt,
                    p.addl, p.ss, theta, biovar, tlag, p.x_r, p.x_i, ode_rtol, ode_atol,
                    max_num_steps, as_rtol, as_atol, as_max_num_steps, nullptr);
  };
}

}  // namespace

TYPED_TEST(pmx_solve_ode_jacobian, transient_schedule) {
  const problem p = transient_problem();
  int count = 0;
  expect_jacobian_path_matches(full_solve<TypeParam>(p),
                               full_solve<TypeParam>(p, counting_jac{&count}), p,
                               1e-6, std::string(TypeParam::name) + " transient");
  EXPECT_GT(count, 0) << "the Jacobian overload never evaluated jac";
}

TYPED_TEST(pmx_solve_ode_jacobian, steady_state_schedule) {
  const problem p = steady_state_problem();
  int count = 0;
  expect_jacobian_path_matches(full_solve<TypeParam>(p),
                               full_solve<TypeParam>(p, counting_jac{&count}), p,
                               1e-6, std::string(TypeParam::name) + " steady state");
  EXPECT_GT(count, 0) << "the Jacobian overload never evaluated jac";
}

/*
 * The shorter overloads: without x_r/x_i, without tolerances, without
 * tlag/biovar, and a right hand side and Jacobian with the stream third.
 */
TYPED_TEST(pmx_solve_ode_jacobian, overloads) {
  const problem p = transient_problem();
  std::vector<std::vector<var>> theta;
  for (const auto& row : p.theta) {
    theta.emplace_back(row.begin(), row.end());
  }
  auto expect_same = [](const Eigen::Matrix<var, -1, -1>& a,
                        const Eigen::Matrix<var, -1, -1>& b, const char* what) {
    ASSERT_EQ(a.size(), b.size());
    for (Eigen::Index i = 0; i < a.size(); ++i) {
      EXPECT_NEAR(a(i).val(), b(i).val(), 1e-6 * (1 + std::abs(a(i).val()))) << what;
    }
  };
  TypeParam solve;
  const twocpt_ir_rhs f;
  const twocpt_ir_jac jac;
  expect_same(solve(f, 4, p.time, p.amt, p.rate, p.ii, p.evid, p.cmt, p.addl, p.ss,
                    theta, p.biovar, p.tlag, ode_rtol, ode_atol, max_num_steps, nullptr),
              solve(f, jac, 4, p.time, p.amt, p.rate, p.ii, p.evid, p.cmt, p.addl,
                    p.ss, theta, p.biovar, p.tlag, ode_rtol, ode_atol, max_num_steps, nullptr),
              "theta, biovar, tlag, tolerances");
  expect_same(solve(f, 4, p.time, p.amt, p.rate, p.ii, p.evid, p.cmt, p.addl, p.ss,
                    theta[0], p.biovar[0], p.tlag[0], ode_rtol, ode_atol,
                    max_num_steps, nullptr),
              solve(f, jac, 4, p.time, p.amt, p.rate, p.ii, p.evid, p.cmt, p.addl,
                    p.ss, theta[0], p.biovar[0], p.tlag[0], ode_rtol, ode_atol,
                    max_num_steps, nullptr),
              "1-d theta, biovar, tlag");
  expect_same(solve(f, 4, p.time, p.amt, p.rate, p.ii, p.evid, p.cmt, p.addl, p.ss,
                    theta[0], ode_rtol, ode_atol, max_num_steps, nullptr),
              solve(f, jac, 4, p.time, p.amt, p.rate, p.ii, p.evid, p.cmt, p.addl,
                    p.ss, theta[0], ode_rtol, ode_atol, max_num_steps, nullptr),
              "1-d theta only");
  expect_same(solve(twocpt_ir_rhs_stream_third(), 4, p.time, p.amt, p.rate, p.ii,
                    p.evid, p.cmt, p.addl, p.ss, theta, p.biovar, p.tlag, p.x_r,
                    p.x_i, ode_rtol, ode_atol, max_num_steps, as_rtol, as_atol,
                    as_max_num_steps, nullptr),
              solve(twocpt_ir_rhs_stream_third(), twocpt_ir_jac_stream_third(), 4,
                    p.time, p.amt, p.rate, p.ii, p.evid, p.cmt, p.addl, p.ss,
                    theta, p.biovar, p.tlag, p.x_r, p.x_i, ode_rtol, ode_atol,
                    max_num_steps, as_rtol, as_atol, as_max_num_steps, nullptr),
              "stream third");
}

/*
 * The Jacobian drives the derivatives: scaling its dka column changes the
 * derivatives with respect to ka (and only those); with
 * STAN_MATH_CHECK_USER_JACOBIAN the scaled Jacobian is refused. A Jacobian
 * of the wrong size is refused in either build.
 */
TYPED_TEST(pmx_solve_ode_jacobian, wrong_jacobian) {
  const problem p = transient_problem();
  int count = 0;
  const counting_jac scaled{&count, 4, 1.5};
  auto solve_d_ka = [&](const auto& jac) {
    stan::math::nested_rev_autodiff nested;
    std::vector<var> theta_v(theta_ref.begin(), theta_ref.end());
    auto y = TypeParam()(twocpt_ir_rhs(), jac, 4, p.time, p.amt, p.rate, p.ii,
                         p.evid, p.cmt, p.addl, p.ss, theta_v, p.biovar, p.tlag,
                         ode_rtol, ode_atol, max_num_steps, nullptr);
    y(1, 10).grad();
    return theta_v[4].adj();
  };
#ifdef STAN_MATH_CHECK_USER_JACOBIAN
  EXPECT_THROW(solve_d_ka(scaled), std::domain_error);
#else
  const double right = solve_d_ka(counting_jac{&count});
  const double wrong = solve_d_ka(scaled);
  EXPECT_GT(std::abs(wrong - right), 1e-3 * std::abs(right));
#endif
  EXPECT_THROW(solve_d_ka(short_jac()), std::invalid_argument);
}

/*
 * theta as data, the event inputs as autodiff variables: the argument
 * columns the solver reads are then only those of the infusion rates.
 */
TYPED_TEST(pmx_solve_ode_jacobian, data_theta) {
  const problem p = transient_problem();
  auto data_theta = [&p](const auto& solve) {
    return [&p, solve](const auto& theta, const auto&... rest) {
      return solve(p.theta, rest...);
    };
  };
  int count = 0;
  expect_jacobian_path_matches(
      data_theta(full_solve<TypeParam>(p)),
      data_theta(full_solve<TypeParam>(p, counting_jac{&count})), p, 1e-6,
      std::string(TypeParam::name) + " data theta", 0.1);
  EXPECT_GT(count, 0) << "the Jacobian overload never evaluated jac";

  // positive control: the state block of jac drives the sensitivities to
  // the doses (scaling d/d central changes d response / d amt)
  auto d_response_d_amt = [&](const auto& jac) {
    stan::math::nested_rev_autodiff nested;
    std::vector<var> amt(p.amt.begin(), p.amt.end());
    auto y = TypeParam()(twocpt_ir_rhs(), jac, 4, p.time, amt, p.rate, p.ii,
                         p.evid, p.cmt, p.addl, p.ss, p.theta, p.biovar, p.tlag,
                         ode_rtol, ode_atol, max_num_steps, nullptr);
    y(3, 10).grad();
    return amt[0].adj();
  };
  const counting_jac scaled{&count, -1, 1.5, 1};
#ifdef STAN_MATH_CHECK_USER_JACOBIAN
  EXPECT_THROW(d_response_d_amt(scaled), std::domain_error);
#else
  const double right = d_response_d_amt(counting_jac{&count});
  const double wrong = d_response_d_amt(scaled);
  EXPECT_GT(std::abs(wrong - right), 1e-3 * std::abs(right));
#endif
}
