#include <stan/math/rev.hpp>
#include <stan/math/torsten/pmx_solve_dde_rk45.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using stan::math::var;

/**
 * y'(t) = -a y(t - tau), theta = (a).
 */
struct lag_decay_rhs {
  template <typename T0, typename T1, typename T2, typename T3>
  Eigen::Matrix<stan::return_type_t<T0, T1, T2, T3>, -1, 1>
  operator()(const T0& t, const Eigen::Matrix<T1, -1, 1>& y,
             const Eigen::Matrix<T2, -1, -1>& y_lag,
             const std::vector<T3>& theta, const std::vector<double>& x_r,
             const std::vector<int>& x_i, std::ostream* msgs) const {
    Eigen::Matrix<stan::return_type_t<T0, T1, T2, T3>, -1, 1> dydt(1);
    dydt(0) = -theta[0] * y_lag(0, 0);
    return dydt;
  }
};

/**
 * Exact solution of y'(t) = -a y(t - tau) with history 0 and a unit dose at
 * t = 0, and its derivatives with respect to a and tau. On
 * [n tau, (n + 1) tau], y(t) = sum_{k=0}^n (-a)^k (t - k tau)^k / k!; on
 * [tau, 2 tau], dy/dtau = a, which only the jump condition at tau gives.
 */
void unit_dose_lag_decay_exact(double t, double a, double tau, double& y,
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
 * One-compartment PK with first-order absorption (depot, central) driving an
 * indirect response R whose production is stimulated by the concentration
 * one delay tau_1 earlier and fed back by R one delay tau_2 earlier:
 *
 *   R' = kin (1 + Smax C(t - tau_1) / (SC50 + C(t - tau_1)))
 *            sqrt(kin / kout / R(t - tau_2)) - kout R,
 *
 * theta = (ka, CL, V, kin, kout, Smax, SC50). A nonempty x_r scales CL by
 * x_r[0]; a nonempty x_i with x_i[0] == 0 turns the feedback off.
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
 * A NONMEM-style event schedule.
 */
struct schedule {
  std::vector<double> time, amt, rate, ii;
  std::vector<int> evid, cmt, addl, ss;

  void add(double t, int ev, int c, double a = 0, double r = 0,
           double i = 0, int ad = 0, int s = 0) {
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
const std::vector<double> tau_ref{2.0, 5.0};
const std::vector<double> y_hist_ref{0.0, 0.0, 7.0};

/**
 * The schedule of pmx_solve_dde_rk45_reference.jl: oral doses into the
 * depot every 24 h, an IV bolus into the central compartment at 30 and an
 * infusion into it at 55, with observations in between.
 */
schedule reference_schedule() {
  schedule s;
  s.add(0, 1, 1, 100, 0, 24, 3);
  s.observe({1.5, 3.3, 7.1, 11.7, 18.2, 26.9});
  s.add(30, 1, 2, 40);
  s.observe({31.4, 36.6, 44.2, 50.8});
  s.add(55, 1, 2, 60, 20);
  s.observe({57.3, 63.9, 75.5, 88.1, 99.4});
  return s;
}

template <typename T>
Eigen::Matrix<T, -1, 1> to_eigen(const std::vector<T>& x) {
  return Eigen::Map<const Eigen::Matrix<T, -1, 1>>(x.data(), x.size());
}

/**
 * base with the entries at the given indices replaced by x.
 */
template <typename T>
std::vector<T> replace(const std::vector<double>& base,
                       const std::vector<int>& indices, const std::vector<T>& x) {
  std::vector<T> res(base.begin(), base.end());
  for (size_t j = 0; j < indices.size(); ++j) {
    res[indices[j]] = x[j];
  }
  return res;
}

/**
 * Compare the reverse-mode gradients of every entry of solve(x) with
 * respect to x with a 6-point central finite difference of step h.
 */
template <typename F>
void expect_gradients_near_fd(const F& solve, const std::vector<double>& x,
                              double h, double tol, const std::string& name) {
  stan::math::nested_rev_autodiff nested;
  std::vector<var> x_v(x.begin(), x.end());
  Eigen::Matrix<var, -1, -1> y = solve(x_v);
  std::vector<Eigen::MatrixXd> fd(x.size());
  for (size_t p = 0; p < x.size(); ++p) {
    auto at = [&](double dx) {
      std::vector<double> x_p = x;
      x_p[p] += dx;
      return Eigen::MatrixXd(solve(x_p));
    };
    fd[p] = (45 * (at(h) - at(-h)) - 9 * (at(2 * h) - at(-2 * h))
             + (at(3 * h) - at(-3 * h)))
            / (60 * h);
  }
  for (Eigen::Index i = 0; i < y.rows(); ++i) {
    for (Eigen::Index k = 0; k < y.cols(); ++k) {
      nested.set_zero_all_adjoints();
      y(i, k).grad();
      for (size_t p = 0; p < x.size(); ++p) {
        EXPECT_NEAR(x_v[p].adj(), fd[p](i, k), tol * (1 + std::abs(fd[p](i, k))))
            << name << "[" << p << "] at (" << i << ", " << k << ")";
      }
    }
  }
}

}  // namespace

/*
 * y' = -a y(t - tau), history 0, unit dose at 0: values and the derivatives
 * with respect to a and tau against the exact solution. On [tau, 2 tau]
 * dy/dtau = a comes only from the jump condition where t - tau crosses the
 * dose.
 */
TEST(pmx_solve_dde_rk45, unit_dose_lag_decay_exact) {
  const double a = 0.8;
  const double tau = 1.0;
  schedule s;
  s.add(0, 1, 1, 1.0);
  const std::vector<double> ts{0.3, 0.7, 1.2, 1.6, 1.9, 2.5, 3.4, 4.1};
  for (double t : ts) {
    s.add(t, 0, 1);
  }
  std::vector<var> theta{a};
  Eigen::Matrix<var, -1, 1> tau_v(1);
  tau_v << tau;
  Eigen::VectorXd y_hist = Eigen::VectorXd::Zero(1);
  Eigen::Matrix<var, -1, -1> y = torsten::pmx_solve_dde_rk45(
      lag_decay_rhs(), 1, s.time, s.amt, s.rate, s.ii, s.evid, s.cmt, s.addl,
      s.ss, theta, tau_v, y_hist, 1e-11, 1e-11, 100000, nullptr);
  ASSERT_EQ(y.rows(), 1);
  ASSERT_EQ(y.cols(), ts.size() + 1);
  EXPECT_FLOAT_EQ(y(0, 0).val(), 1.0);
  for (size_t i = 0; i < ts.size(); ++i) {
    double y_exact, dy_da, dy_dtau;
    unit_dose_lag_decay_exact(ts[i], a, tau, y_exact, dy_da, dy_dtau);
    stan::math::set_zero_all_adjoints();
    y(0, i + 1).grad();
    EXPECT_NEAR(y(0, i + 1).val(), y_exact, 1e-9) << "t = " << ts[i];
    EXPECT_NEAR(theta[0].adj(), dy_da, 1e-8) << "t = " << ts[i];
    EXPECT_NEAR(tau_v(0).adj(), dy_dtau, 1e-8) << "t = " << ts[i];
  }
  stan::math::recover_memory();
}

/*
 * The delayed indirect-response model against DelayDiffEq.jl (method of steps
 * over Vern9, tolerances 1e-13; pmx_solve_dde_rk45_reference.jl): two
 * delays, a history that is not the steady state, oral doses with ADDL, an
 * IV bolus whose concentration jump reaches the response through the delay,
 * and an infusion.
 */
TEST(pmx_solve_dde_rk45, delayed_indirect_response_reference) {
  const schedule s = reference_schedule();
  Eigen::MatrixXd reference(15, 3);
  reference <<
    25.92402606458916, 65.67931867681061, 7.485439534245756,  // t = 1.5
    5.130331033191917, 69.58171300123043, 9.222604243956773,  // t = 3.3
    0.16782562013892904, 45.35460645047423, 12.781510697130884,  // t = 7.1
    0.002672262428125171, 24.665010485066123, 11.644839419953422,  // t = 11.7
    7.695846316947785e-6, 10.369256247773505, 9.435636973797749,  // t = 18.2
    7.3534543793658464, 74.3642723248416, 9.22551474523345,  // t = 26.9
    0.1281146370917427, 78.58790853967994, 12.267551475830201,  // t = 31.4
    0.0011887774967105382, 39.360428475457255, 12.112676638067244,  // t = 36.6
    1.2721148406461662e-6, 14.288613690111939, 9.942846284679241,  // t = 44.2
    8.045960678301663, 77.29778188549018, 9.454665633346039,  // t = 50.8
    0.02317155538345324, 76.05089005982475, 12.208483387020634,  // t = 57.3
    6.0988220829353236e-5, 37.64240425465628, 11.66421544167033,  // t = 63.9
    4.285212688487495, 76.60055405539745, 10.264383991545248,  // t = 75.5
    5.094164410573206e-5, 15.213838174068464, 10.126599088947108,  // t = 88.1
    1.9511833279093684e-9, 3.3721235627333592, 8.583243481573554;  // t = 99.4
  const std::vector<int> observations{1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13,
                                      14, 15, 16, 17};

  Eigen::MatrixXd y = torsten::pmx_solve_dde_rk45(
      delayed_indirect_response_rhs(), 3, s.time, s.amt, s.rate, s.ii, s.evid,
      s.cmt, s.addl, s.ss, theta_ref, to_eigen(tau_ref), to_eigen(y_hist_ref),
      1e-10, 1e-10, 100000, nullptr);
  ASSERT_EQ(y.cols(), s.time.size());
  for (size_t j = 0; j < observations.size(); ++j) {
    for (int i = 0; i < 3; ++i) {
      EXPECT_NEAR(y(i, observations[j]), reference(j, i),
                  1e-8 * (1 + std::abs(reference(j, i))))
          << "compartment " << i << " at t = " << s.time[observations[j]];
    }
  }
}

/*
 * Gradients of the delayed indirect-response model with respect to every
 * input that can be a parameter against finite differences, on a schedule
 * with a lag time on the oral doses, bioavailabilities, a parameter row that
 * changes at the IV bolus, and an infusion. The event times move the
 * observations, the doses (and with them their ADDL doses, lagged doses and
 * the end of the infusion) and the start of the integration.
 */
TEST(pmx_solve_dde_rk45, delayed_indirect_response_gradients) {
  const schedule s = reference_schedule();
  const size_t n = s.time.size();
  std::vector<std::vector<double>> theta(n, theta_ref);
  for (size_t i = 8; i < n; ++i) {
    theta[i][1] = 5.5;  // CL changes at the IV bolus
  }
  const std::vector<double> biovar{0.8, 0.95, 1.0};
  // lag times on the oral doses and on the IV bolus, whose concentration jump
  // reaches the response through the delay
  const std::vector<double> tlag{0.7, 0.4, 0.0};
  const double rtol = 1e-11;

  auto solve = [&](const auto& theta_, const auto& tau_, const auto& y_hist_,
                   const auto& amt_, const auto& rate_, const auto& ii_,
                   const auto& biovar_, const auto& tlag_) {
    return torsten::pmx_solve_dde_rk45(
        delayed_indirect_response_rhs(), 3, s.time, amt_, rate_, ii_, s.evid,
        s.cmt, s.addl, s.ss, theta_, to_eigen(tau_), to_eigen(y_hist_),
        biovar_, tlag_, rtol, rtol, 1000000, nullptr);
  };
  const double h = 1e-3;
  const double tol = 1e-6;
  // the dosing events: oral (with ADDL), IV bolus, infusion
  const std::vector<int> doses{0, 7, 12};

  expect_gradients_near_fd(
      [&](const auto& x) {
        using T = typename std::decay_t<decltype(x)>::value_type;
        std::vector<std::vector<T>> th(n);
        for (size_t i = 0; i < n; ++i) {
          // the rows share one variable per parameter, except CL from the
          // bolus on, which is its own
          th[i] = {x[0], i < 8 ? x[1] : x[7], x[2], x[3], x[4], x[5], x[6]};
        }
        return solve(th, tau_ref, y_hist_ref, s.amt, s.rate, s.ii, biovar, tlag);
      },
      {0.9, 4.0, 30.0, 2.0, 0.25, 1.5, 2.0, 5.5}, h, tol, "theta");
  expect_gradients_near_fd(
      [&](const auto& x) {
        return solve(theta, x, y_hist_ref, s.amt, s.rate, s.ii, biovar, tlag);
      },
      tau_ref, h, tol, "tau");
  expect_gradients_near_fd(
      [&](const auto& x) {
        return solve(theta, tau_ref, x, s.amt, s.rate, s.ii, biovar, tlag);
      },
      {0.5, 1.0, 7.0}, h, tol, "y_hist");
  expect_gradients_near_fd(
      [&](const auto& x) {
        return solve(theta, tau_ref, y_hist_ref, replace(s.amt, doses, x),
                     s.rate, s.ii, biovar, tlag);
      },
      {100.0, 40.0, 60.0}, 0.5, tol, "amt");
  expect_gradients_near_fd(
      [&](const auto& x) {
        return solve(theta, tau_ref, y_hist_ref, s.amt,
                     replace(s.rate, {12}, x), s.ii, biovar, tlag);
      },
      {20.0}, 1e-2, tol, "rate");
  expect_gradients_near_fd(
      [&](const auto& x) {
        return solve(theta, tau_ref, y_hist_ref, s.amt, s.rate,
                     replace(s.ii, {0}, x), biovar, tlag);
      },
      {24.0}, 1e-2, tol, "ii");
  expect_gradients_near_fd(
      [&](const auto& x) {
        return solve(theta, tau_ref, y_hist_ref, s.amt, s.rate, s.ii, x, tlag);
      },
      biovar, h, tol, "biovar");
  expect_gradients_near_fd(
      [&](const auto& x) {
        return solve(theta, tau_ref, y_hist_ref, s.amt, s.rate, s.ii, biovar,
                     replace(tlag, {0, 1}, x));
      },
      {0.7, 0.4}, h, tol, "tlag");
  expect_gradients_near_fd(
      [&](const auto& x) {
        return torsten::pmx_solve_dde_rk45(
            delayed_indirect_response_rhs(), 3, x, s.amt, s.rate, s.ii, s.evid,
            s.cmt, s.addl, s.ss, theta, to_eigen(tau_ref), to_eigen(y_hist_ref),
            biovar, tlag, rtol, rtol, 1000000, nullptr);
      },
      s.time, h, tol, "time");
}

/*
 * Gradients with respect to the event times through a turn-off (EVID 2,
 * negative CMT), whose jump depends on the state, a reset (EVID 3), after
 * which the solution depends on the reset time, and a reset with a dose
 * (EVID 4).
 */
TEST(pmx_solve_dde_rk45, event_time_gradients) {
  schedule s;
  s.add(0.5, 1, 2, 50);
  s.observe({3.1, 8.2});
  s.add(10, 2, -2);
  s.observe({12.5});
  s.add(20, 3, 2);
  s.observe({24.4});
  s.add(30, 4, 2, 40);
  s.observe({33.3, 41.7});
  expect_gradients_near_fd(
      [&](const auto& x) {
        return torsten::pmx_solve_dde_rk45(
            delayed_indirect_response_rhs(), 3, x, s.amt, s.rate, s.ii, s.evid,
            s.cmt, s.addl, s.ss, theta_ref, to_eigen(tau_ref),
            to_eigen(y_hist_ref), 1e-11, 1e-11, 1000000, nullptr);
      },
      s.time, 1e-3, 1e-6, "time");
}

/*
 * A reset (EVID 3) and a reset with a dose (EVID 4) restart from the
 * history: the solution after them, and its gradients, are those of a
 * schedule that starts there. A compartment turned off (EVID 2, negative
 * CMT) is set to zero.
 */
TEST(pmx_solve_dde_rk45, reset_restarts_from_history) {
  schedule full;
  full.add(0, 1, 2, 50);
  full.observe({3.1, 8.2});
  full.add(10, 2, -2);
  full.observe({12.5});
  full.add(20, 3, 2);
  full.observe({24.4});
  full.add(30, 4, 2, 40);
  full.observe({33.3, 41.7});

  schedule tail;
  tail.add(30, 1, 2, 40);
  tail.observe({33.3, 41.7});

  auto solve = [&](const schedule& s, const auto& theta, const auto& tau,
                   const auto& y_hist) {
    return torsten::pmx_solve_dde_rk45(
        delayed_indirect_response_rhs(), 3, s.time, s.amt, s.rate, s.ii, s.evid,
        s.cmt, s.addl, s.ss, theta, to_eigen(tau), to_eigen(y_hist), 1e-10,
        1e-10, 100000, nullptr);
  };

  stan::math::nested_rev_autodiff nested;
  std::vector<var> theta(theta_ref.begin(), theta_ref.end());
  std::vector<var> tau(tau_ref.begin(), tau_ref.end());
  std::vector<var> y_hist(y_hist_ref.begin(), y_hist_ref.end());
  Eigen::Matrix<var, -1, -1> y_full = solve(full, theta, tau, y_hist);
  Eigen::Matrix<var, -1, -1> y_tail = solve(tail, theta, tau, y_hist);

  // turned off at 10
  EXPECT_EQ(y_full(1, 3).val(), 0.0);
  // reset at 20: the history
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(y_full(i, 5).val(), y_hist_ref[i]);
  }
  std::vector<var> params;
  params.insert(params.end(), theta.begin(), theta.end());
  params.insert(params.end(), tau.begin(), tau.end());
  params.insert(params.end(), y_hist.begin(), y_hist.end());
  for (int k = 0; k < 3; ++k) {
    for (int i = 0; i < 3; ++i) {
      var a = y_full(i, 7 + k);
      var b = y_tail(i, k);
      EXPECT_NEAR(a.val(), b.val(), 1e-12 * (1 + std::abs(b.val())));
      nested.set_zero_all_adjoints();
      a.grad();
      std::vector<double> ga;
      for (const var& p : params) {
        ga.push_back(p.adj());
      }
      nested.set_zero_all_adjoints();
      b.grad();
      for (size_t p = 0; p < params.size(); ++p) {
        EXPECT_NEAR(ga[p], params[p].adj(), 1e-10 * (1 + std::abs(ga[p])))
            << "parameter " << p << " at (" << i << ", " << k << ")";
      }
    }
  }
}

/*
 * Real and integer data rows per event reach the right hand side of their
 * interval: scaling CL through x_r after the bolus gives the solution of
 * the corresponding parameter rows.
 */
TEST(pmx_solve_dde_rk45, data_rows) {
  const schedule s = reference_schedule();
  const size_t n = s.time.size();
  std::vector<std::vector<double>> x_r(n, {1.0});
  std::vector<std::vector<double>> theta(n, theta_ref);
  for (size_t i = 8; i < n; ++i) {
    x_r[i][0] = 1.5;
    theta[i][1] *= 1.5;
  }
  const std::vector<std::vector<int>> x_i(n, {1});
  const std::vector<double> biovar(3, 1.0), tlag(3, 0.0);
  const Eigen::VectorXd tau = to_eigen(tau_ref), y_hist = to_eigen(y_hist_ref);

  Eigen::MatrixXd y_data = torsten::pmx_solve_dde_rk45(
      delayed_indirect_response_rhs(), 3, s.time, s.amt, s.rate, s.ii, s.evid,
      s.cmt, s.addl, s.ss, theta_ref, tau, y_hist, biovar, tlag, x_r, x_i,
      1e-10, 1e-10, 100000, nullptr);
  Eigen::MatrixXd y_theta = torsten::pmx_solve_dde_rk45(
      delayed_indirect_response_rhs(), 3, s.time, s.amt, s.rate, s.ii, s.evid,
      s.cmt, s.addl, s.ss, theta, tau, y_hist, 1e-10, 1e-10, 100000, nullptr);
  EXPECT_LT((y_data - y_theta).cwiseAbs().maxCoeff(), 1e-12);

  // x_i = 0 turns the feedback off
  const std::vector<std::vector<int>> no_feedback(1, {0});
  Eigen::MatrixXd y_off = torsten::pmx_solve_dde_rk45(
      delayed_indirect_response_rhs(), 3, s.time, s.amt, s.rate, s.ii, s.evid,
      s.cmt, s.addl, s.ss, theta_ref, tau, y_hist, biovar, tlag, x_r,
      no_feedback, 1e-10, 1e-10, 100000, nullptr);
  EXPECT_GT((y_off.row(2) - y_data.row(2)).cwiseAbs().maxCoeff(), 0.1);
  EXPECT_LT((y_off.topRows(2) - y_data.topRows(2)).cwiseAbs().maxCoeff(), 1e-7);
}

/*
 * The optional arguments take their documented defaults: bioavailability 1,
 * no lag time, no data, tolerances 1e-6 and step limits 1e6 / 100.
 */
TEST(pmx_solve_dde_rk45, overloads) {
  const schedule s = reference_schedule();
  const Eigen::VectorXd tau = to_eigen(tau_ref), y_hist = to_eigen(y_hist_ref);
  const std::vector<double> biovar(3, 1.0), tlag(3, 0.0);
  const std::vector<std::vector<double>> theta_2d(1, theta_ref), biovar_2d(1, biovar),
      tlag_2d(1, tlag), x_r(1, std::vector<double>());
  const std::vector<std::vector<int>> x_i(1, std::vector<int>());
  delayed_indirect_response_rhs f;

  Eigen::MatrixXd y0 = torsten::pmx_solve_dde_rk45(
      f, 3, s.time, s.amt, s.rate, s.ii, s.evid, s.cmt, s.addl, s.ss, theta_2d,
      tau, y_hist, biovar_2d, tlag_2d, x_r, x_i, 1e-6, 1e-6, 1e6, 1e-6, 1e-6,
      100, nullptr);
#define EXPECT_SAME_SOLUTION(...)                                           \
  {                                                                         \
    Eigen::MatrixXd y = torsten::pmx_solve_dde_rk45(                        \
        f, 3, s.time, s.amt, s.rate, s.ii, s.evid, s.cmt, s.addl, s.ss,     \
        __VA_ARGS__);                                                       \
    EXPECT_TRUE(y == y0) << #__VA_ARGS__;                                   \
  }
  EXPECT_SAME_SOLUTION(theta_ref, tau, y_hist, nullptr);
  EXPECT_SAME_SOLUTION(theta_2d, tau, y_hist, nullptr);
  EXPECT_SAME_SOLUTION(theta_ref, tau, y_hist, 1e-6, 1e-6, 1e6, nullptr);
  EXPECT_SAME_SOLUTION(theta_2d, tau, y_hist, 1e-6, 1e-6, 1e6, 1e-6, 1e-6, 100,
                       nullptr);
  EXPECT_SAME_SOLUTION(theta_ref, tau, y_hist, biovar, nullptr);
  EXPECT_SAME_SOLUTION(theta_2d, tau, y_hist, biovar_2d, 1e-6, 1e-6, 1e6,
                       nullptr);
  EXPECT_SAME_SOLUTION(theta_ref, tau, y_hist, biovar, tlag, nullptr);
  EXPECT_SAME_SOLUTION(theta_ref, tau, y_hist, biovar_2d, tlag, 1e-6, 1e-6,
                       1e6, 1e-6, 1e-6, 100, nullptr);
  EXPECT_SAME_SOLUTION(theta_2d, tau, y_hist, biovar, tlag_2d, x_r, nullptr);
  EXPECT_SAME_SOLUTION(theta_ref, tau, y_hist, biovar, tlag, x_r, 1e-6, 1e-6,
                       1e6, nullptr);
  EXPECT_SAME_SOLUTION(theta_ref, tau, y_hist, biovar, tlag, x_r, x_i, nullptr);
  EXPECT_SAME_SOLUTION(theta_ref, tau, y_hist, biovar, tlag, x_r, x_i, 1e-6,
                       1e-6, 1e6, nullptr);
#undef EXPECT_SAME_SOLUTION
}

TEST(pmx_solve_dde_rk45, argument_errors) {
  schedule s = reference_schedule();
  const Eigen::VectorXd tau = to_eigen(tau_ref), y_hist = to_eigen(y_hist_ref);
  delayed_indirect_response_rhs f;
  auto solve = [&](const schedule& s, const Eigen::VectorXd& tau,
                   const Eigen::VectorXd& y_hist) {
    return torsten::pmx_solve_dde_rk45(f, 3, s.time, s.amt, s.rate, s.ii,
                                       s.evid, s.cmt, s.addl, s.ss, theta_ref,
                                       tau, y_hist, nullptr);
  };
  EXPECT_NO_THROW(solve(s, tau, y_hist));

  Eigen::VectorXd tau_bad = tau;
  tau_bad(1) = 0;
  EXPECT_THROW(solve(s, tau_bad, y_hist), std::domain_error);
  tau_bad(1) = -1;
  EXPECT_THROW(solve(s, tau_bad, y_hist), std::domain_error);
  EXPECT_THROW(solve(s, tau, Eigen::VectorXd::Zero(2)), std::invalid_argument);
  Eigen::VectorXd y_hist_bad = y_hist;
  y_hist_bad(2) = std::numeric_limits<double>::infinity();
  EXPECT_THROW(solve(s, tau, y_hist_bad), std::domain_error);

  schedule bad_cmt = s;
  bad_cmt.cmt[0] = 4;
  EXPECT_THROW(solve(bad_cmt, tau, y_hist), std::domain_error);

  // refused: adding a steady state to the current state relies on
  // superposition, which a nonlinear DDE (and its history) does not have
  // (decision Torsten:dde 0cp4xgg)
  schedule overlay = s;
  overlay.ss[0] = 2;
  overlay.addl[0] = 0;
  EXPECT_THROW(solve(overlay, tau, y_hist), std::domain_error);
}

namespace {

/**
 * Value and gradient with respect to params of two results agree.
 */
void expect_same_solution(Eigen::Matrix<var, -1, -1>& a, int col_a,
                          Eigen::Matrix<var, -1, -1>& b, int col_b,
                          std::vector<var>& params, double tol,
                          const std::string& name) {
  for (int i = 0; i < a.rows(); ++i) {
    EXPECT_NEAR(a(i, col_a).val(), b(i, col_b).val(),
                tol * (1 + std::abs(b(i, col_b).val())))
        << name << ": value of compartment " << i;
    stan::math::set_zero_all_adjoints_nested();
    a(i, col_a).grad();
    std::vector<double> ga;
    for (const var& p : params) {
      ga.push_back(p.adj());
    }
    stan::math::set_zero_all_adjoints_nested();
    b(i, col_b).grad();
    for (size_t p = 0; p < params.size(); ++p) {
      EXPECT_NEAR(ga[p], params[p].adj(), tol * (1 + std::abs(params[p].adj())))
          << name << ": derivative of compartment " << i << " w.r.t. parameter " << p;
    }
  }
}

}  // namespace

/*
 * A steady-state dose (SS = 1, II > 0) restarts from the steady state of its
 * regimen: the solution after it, and its gradients, agree with a long
 * transient run of the same regimen from the history. For an oral bolus
 * regimen, and for an IV infusion regimen.
 */
TEST(pmx_solve_dde_rk45, periodic_steady_state_matches_long_run) {
  struct regimen {
    int cmt;
    double amt, rate, ii;
  };
  for (const regimen& r : {regimen{1, 100, 0, 24}, regimen{2, 60, 20, 12}}) {
    const std::string name = r.rate > 0 ? "infusion" : "bolus";
    const std::vector<double> ts{201.3, 210.7, 230.2, 247.9, 260.4};
    schedule ss;
    ss.add(200, 1, r.cmt, r.amt, r.rate, r.ii, 6, 1);
    ss.observe(ts);
    const int periods = 80;
    schedule transient;
    transient.add(200 - periods * r.ii, 1, r.cmt, r.amt, r.rate, r.ii, periods + 6);
    transient.observe(ts);

    stan::math::nested_rev_autodiff nested;
    std::vector<var> theta(theta_ref.begin(), theta_ref.end());
    std::vector<var> tau(tau_ref.begin(), tau_ref.end());
    auto solve = [&](const schedule& s) {
      return torsten::pmx_solve_dde_rk45(
          delayed_indirect_response_rhs(), 3, s.time, s.amt, s.rate, s.ii,
          s.evid, s.cmt, s.addl, s.ss, theta, to_eigen(tau),
          to_eigen(y_hist_ref), 1e-10, 1e-10, 10000000, 1e-11, 1e-11, 1000,
          nullptr);
    };
    Eigen::Matrix<var, -1, -1> y_ss = solve(ss);
    Eigen::Matrix<var, -1, -1> y_transient = solve(transient);
    std::vector<var> params(theta);
    params.insert(params.end(), tau.begin(), tau.end());
    for (size_t k = 1; k <= ts.size(); ++k) {
      expect_same_solution(y_ss, k, y_transient, k, params, 1e-7,
                           name + " at t = " + std::to_string(ts[k - 1]));
    }
  }
}

/*
 * A constant-infusion steady state (SS = 1, II = 0) is the equilibrium the
 * infusion approaches: it agrees, with its gradients, with the state after a
 * long infusion.
 */
TEST(pmx_solve_dde_rk45, constant_infusion_steady_state_matches_long_infusion) {
  stan::math::nested_rev_autodiff nested;
  std::vector<var> theta(theta_ref.begin(), theta_ref.end());
  std::vector<var> tau(tau_ref.begin(), tau_ref.end());
  var rate = 5.0;

  schedule ss;
  ss.add(100, 1, 2, 0, 0, 0, 0, 1);
  std::vector<var> ss_rate{rate};
  schedule infusion;
  infusion.add(-1900, 1, 2, 5.0 * 2100);
  infusion.observe({100});
  std::vector<var> infusion_rate{rate, 0.0};

  auto solve = [&](const schedule& s, const std::vector<var>& rate) {
    return torsten::pmx_solve_dde_rk45(
        delayed_indirect_response_rhs(), 3, s.time, s.amt, rate, s.ii, s.evid,
        s.cmt, s.addl, s.ss, theta, to_eigen(tau), to_eigen(y_hist_ref),
        1e-10, 1e-10, 10000000, 1e-12, 1e-12, 1000, nullptr);
  };
  Eigen::Matrix<var, -1, -1> y_ss = solve(ss, ss_rate);
  Eigen::Matrix<var, -1, -1> y_infusion = solve(infusion, infusion_rate);
  std::vector<var> params(theta);
  params.insert(params.end(), tau.begin(), tau.end());
  params.push_back(rate);
  expect_same_solution(y_ss, 0, y_infusion, 1, params, 1e-7, "equilibrium");
}

/*
 * Gradients through the burn-in of steady-state regimens against finite
 * differences, including the dosing interval, the lag time (which moves the
 * steady-state dose), the infusion rate and the event times.
 */
TEST(pmx_solve_dde_rk45, steady_state_gradients) {
  schedule s;
  s.add(0, 1, 1, 100, 0, 24, 2, 1);
  s.observe({3.3, 11.7, 30.2, 55.1});
  s.add(60, 1, 2, 60, 20, 12, 1, 1);
  s.observe({61.4, 66.2, 75.3, 80.0});
  s.add(90, 1, 2, 0, 5, 0, 0, 1);
  s.observe({91.7, 99.9});
  const std::vector<double> biovar{0.8, 0.95, 1.0};
  const std::vector<double> tlag{0.7, 0.0, 0.0};

  auto solve = [&](const auto& theta_, const auto& tau_, const auto& amt_,
                   const auto& rate_, const auto& ii_, const auto& biovar_,
                   const auto& tlag_) {
    return torsten::pmx_solve_dde_rk45(
        delayed_indirect_response_rhs(), 3, s.time, amt_, rate_, ii_, s.evid,
        s.cmt, s.addl, s.ss, theta_, to_eigen(tau_), to_eigen(y_hist_ref),
        biovar_, tlag_, 1e-11, 1e-11, 10000000, 1e-12, 1e-12, 1000, nullptr);
  };
  const double h = 1e-3;
  const double tol = 1e-6;
  expect_gradients_near_fd(
      [&](const auto& x) {
        return solve(x, tau_ref, s.amt, s.rate, s.ii, biovar, tlag);
      },
      theta_ref, h, tol, "theta");
  expect_gradients_near_fd(
      [&](const auto& x) {
        return solve(theta_ref, x, s.amt, s.rate, s.ii, biovar, tlag);
      },
      tau_ref, h, tol, "tau");
  expect_gradients_near_fd(
      [&](const auto& x) {
        return solve(theta_ref, tau_ref, replace(s.amt, {0, 5}, x), s.rate,
                     s.ii, biovar, tlag);
      },
      {100.0, 60.0}, 0.5, tol, "amt");
  expect_gradients_near_fd(
      [&](const auto& x) {
        return solve(theta_ref, tau_ref, s.amt, replace(s.rate, {5, 10}, x),
                     s.ii, biovar, tlag);
      },
      {20.0, 5.0}, 1e-2, tol, "rate");
  expect_gradients_near_fd(
      [&](const auto& x) {
        return solve(theta_ref, tau_ref, s.amt, s.rate, replace(s.ii, {0, 5}, x),
                     biovar, tlag);
      },
      {24.0, 12.0}, 1e-2, tol, "ii");
  expect_gradients_near_fd(
      [&](const auto& x) {
        return solve(theta_ref, tau_ref, s.amt, s.rate, s.ii, x, tlag);
      },
      biovar, h, tol, "biovar");
  expect_gradients_near_fd(
      [&](const auto& x) {
        return solve(theta_ref, tau_ref, s.amt, s.rate, s.ii, biovar,
                     replace(tlag, {0}, x));
      },
      {0.7}, h, tol, "tlag");
  expect_gradients_near_fd(
      [&](const auto& x) {
        return torsten::pmx_solve_dde_rk45(
            delayed_indirect_response_rhs(), 3, x, s.amt, s.rate, s.ii, s.evid,
            s.cmt, s.addl, s.ss, theta_ref, to_eigen(tau_ref),
            to_eigen(y_hist_ref), biovar, tlag, 1e-11, 1e-11, 10000000, 1e-12,
            1e-12, 1000, nullptr);
      },
      s.time, h, tol, "time");
}

TEST(pmx_solve_dde_rk45, steady_state_errors) {
  const Eigen::VectorXd tau = to_eigen(tau_ref), y_hist = to_eigen(y_hist_ref);
  auto solve = [&](const schedule& s, int as_max_num_steps) {
    return torsten::pmx_solve_dde_rk45(
        delayed_indirect_response_rhs(), 3, s.time, s.amt, s.rate, s.ii,
        s.evid, s.cmt, s.addl, s.ss, theta_ref, tau, y_hist, 1e-8, 1e-8,
        100000, 1e-10, 1e-10, as_max_num_steps, nullptr);
  };
  schedule periodic;
  periodic.add(0, 1, 1, 100, 0, 24, 0, 1);
  periodic.observe({5.0});
  EXPECT_NO_THROW(solve(periodic, 100));
  // the burn-in reports a steady state it has not reached
  EXPECT_THROW(solve(periodic, 2), std::domain_error);

  schedule bolus_no_interval;
  bolus_no_interval.add(0, 1, 1, 100, 0, 0, 0, 1);
  EXPECT_THROW(solve(bolus_no_interval, 100), std::domain_error);

  schedule infusion_with_amount;
  infusion_with_amount.add(0, 1, 2, 10, 5, 0, 0, 1);
  EXPECT_THROW(solve(infusion_with_amount, 100), std::domain_error);
}
