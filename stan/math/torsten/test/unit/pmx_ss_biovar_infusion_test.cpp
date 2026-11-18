#include <stan/math.hpp>
#include <stan/math/rev/core.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

using stan::math::var;

/*
 * A RATE-defined infusion with bioavailability F keeps its rate RATE and
 * lasts F * AMT / RATE (NONMEM's convention), in transient and
 * steady-state dosing alike. So the steady state of a regimen (SS = 1 or
 * SS = 2) is the limit of the transient regimen, the same dose repeated
 * from long ago. Before, the transient path infused F * RATE for
 * AMT / RATE while the steady state infused RATE for F * AMT / RATE, so
 * the two disagreed for F != 1.
 *
 * Every solver is checked: the analytical onecpt and twocpt models,
 * linode, and the rk45 and bdf ODE solvers.
 */

namespace {

const double ode_tol = 1e-10;

/*
 * Each solver functor solves one schedule with a given bioavailability;
 * its compartment 1 is the depot and compartment 2 the central one.
 */
struct onecpt_solver {
  static constexpr int ncmt = 2;
  static constexpr double rtol = 1e-9;
  template <typename T1, typename T2, typename T3, typename T5>
  auto operator()(const std::vector<double>& time, const std::vector<T1>& amt,
                  const std::vector<T2>& rate, const std::vector<T3>& ii,
                  const std::vector<int>& evid, const std::vector<int>& cmt,
                  const std::vector<int>& addl, const std::vector<int>& ss,
                  const std::vector<std::vector<T5>>& biovar) const {
    std::vector<std::vector<double>> theta{{10, 80, 1.2}};  // CL, V, ka
    std::vector<std::vector<double>> tlag{{0, 0}};
    return torsten::pmx_solve_onecpt(time, amt, rate, ii, evid, cmt, addl, ss,
                                     theta, biovar, tlag);
  }
};

struct twocpt_solver {
  static constexpr int ncmt = 3;
  static constexpr double rtol = 1e-9;
  template <typename T1, typename T2, typename T3, typename T5>
  auto operator()(const std::vector<double>& time, const std::vector<T1>& amt,
                  const std::vector<T2>& rate, const std::vector<T3>& ii,
                  const std::vector<int>& evid, const std::vector<int>& cmt,
                  const std::vector<int>& addl, const std::vector<int>& ss,
                  const std::vector<std::vector<T5>>& biovar) const {
    std::vector<std::vector<double>> theta{{5, 8, 20, 70, 1.2}};  // CL, Q, V2, V3, ka
    std::vector<std::vector<double>> tlag{{0, 0, 0}};
    return torsten::pmx_solve_twocpt(time, amt, rate, ii, evid, cmt, addl, ss,
                                     theta, biovar, tlag);
  }
};

struct linode_solver {
  static constexpr int ncmt = 2;
  static constexpr double rtol = 1e-9;
  template <typename T1, typename T2, typename T3, typename T5>
  auto operator()(const std::vector<double>& time, const std::vector<T1>& amt,
                  const std::vector<T2>& rate, const std::vector<T3>& ii,
                  const std::vector<int>& evid, const std::vector<int>& cmt,
                  const std::vector<int>& addl, const std::vector<int>& ss,
                  const std::vector<std::vector<T5>>& biovar) const {
    // the onecpt model as a linear system: ka = 1.2, k10 = CL / V = 0.125
    Eigen::MatrixXd K(2, 2);
    K << -1.2, 0, 1.2, -0.125;
    std::vector<Eigen::MatrixXd> system{K};
    std::vector<std::vector<double>> tlag{{0, 0}};
    return torsten::pmx_solve_linode(time, amt, rate, ii, evid, cmt, addl, ss,
                                     system, biovar, tlag);
  }
};

struct rk45_solver {
  static constexpr int ncmt = 2;
  static constexpr double rtol = 1e-6;
  template <typename T1, typename T2, typename T3, typename T5>
  auto operator()(const std::vector<double>& time, const std::vector<T1>& amt,
                  const std::vector<T2>& rate, const std::vector<T3>& ii,
                  const std::vector<int>& evid, const std::vector<int>& cmt,
                  const std::vector<int>& addl, const std::vector<int>& ss,
                  const std::vector<std::vector<T5>>& biovar) const {
    std::vector<double> theta{10, 80, 1.2};
    std::vector<std::vector<double>> tlag{{0, 0}};
    return torsten::pmx_solve_rk45(torsten::PMXOneCptODE(), ncmt, time, amt, rate, ii,
                                   evid, cmt, addl, ss, theta, biovar, tlag,
                                   ode_tol, ode_tol, 1000000, ode_tol, ode_tol, 1000,
                                   nullptr);
  }
};

struct bdf_solver {
  static constexpr int ncmt = 3;
  static constexpr double rtol = 1e-6;
  template <typename T1, typename T2, typename T3, typename T5>
  auto operator()(const std::vector<double>& time, const std::vector<T1>& amt,
                  const std::vector<T2>& rate, const std::vector<T3>& ii,
                  const std::vector<int>& evid, const std::vector<int>& cmt,
                  const std::vector<int>& addl, const std::vector<int>& ss,
                  const std::vector<std::vector<T5>>& biovar) const {
    std::vector<double> theta{5, 8, 20, 70, 1.2};
    std::vector<std::vector<double>> tlag{{0, 0, 0}};
    return torsten::pmx_solve_bdf(torsten::PMXTwoCptODE(), ncmt, time, amt, rate, ii,
                                  evid, cmt, addl, ss, theta, biovar, tlag,
                                  ode_tol, ode_tol, 1000000, ode_tol, ode_tol, 1000,
                                  nullptr);
  }
};

/*
 * A dosing regimen observed at obs_times >= 0: AMT at RATE every II into
 * compartment cmt. steady(ss) doses it with the SS flag at t = 0;
 * transient() starts the same dose 600 h earlier without SS. Both keep
 * dosing past the last observation.
 */
struct regimen {
  double amt, rate, ii;
  int cmt;
  std::vector<double> obs_times{0, 3, 7, 12, 17, 30};

  struct schedule {
    std::vector<double> time, amt, rate, ii;
    std::vector<int> evid, cmt, addl, ss;
  };

  schedule events(double t_dose, int addl, int ss) const {
    schedule s;
    s.time.push_back(t_dose);
    s.amt.push_back(amt);
    s.rate.push_back(rate);
    s.ii.push_back(ii);
    s.evid.push_back(1);
    s.cmt.push_back(cmt);
    s.addl.push_back(addl);
    s.ss.push_back(ss);
    for (double t : obs_times) {
      s.time.push_back(t);
      s.amt.push_back(0);
      s.rate.push_back(0);
      s.ii.push_back(0);
      s.evid.push_back(0);
      s.cmt.push_back(cmt);
      s.addl.push_back(0);
      s.ss.push_back(0);
    }
    return s;
  }

  schedule steady(int ss) const { return events(0, 10, ss); }
  schedule transient() const { return events(-600, 60, 0); }
};

template <typename S>
Eigen::MatrixXd solve(const S& solver, const regimen::schedule& s,
                      const std::vector<std::vector<double>>& biovar) {
  return solver(s.time, s.amt, s.rate, s.ii, s.evid, s.cmt, s.addl, s.ss, biovar);
}

std::vector<std::vector<double>> biovar_cases(int ncmt) {
  std::vector<double> f1(ncmt, 1.0);
  std::vector<double> f{0.8, 0.6, 0.9};
  f.resize(ncmt);
  return {f1, f};
}

/*
 * The SS solution equals that of the long transient run, for F = 1 and
 * F != 1, doses into the depot and the central compartment, infusions
 * shorter and longer than II, and SS = 1 and 2. An infusion longer than
 * II is compared at the dose time only: after an SS dose Torsten does not
 * continue the earlier doses' infusions still running then, whatever F.
 */
template <typename S>
void check_steady_state_is_transient_limit(const S& solver) {
  for (const std::vector<double>& f : biovar_cases(S::ncmt)) {
    const std::vector<std::vector<double>> biovar{f};
    for (int cmt : {1, 2}) {
      for (double rate : {150.0, 50.0}) {  // 8 h (< II) and 24 h (> II) at F = 1
        const regimen r{1200, rate, 12, cmt};
        const Eigen::MatrixXd tr = solve(solver, r.transient(), biovar);
        for (int ss : {1, 2}) {
          const Eigen::MatrixXd st = solve(solver, r.steady(ss), biovar);
          const bool overlapping = f[cmt - 1] * r.amt / r.rate > r.ii;
          for (size_t k = 0; k < (overlapping ? 1 : r.obs_times.size()); ++k) {
            for (int j = 0; j < S::ncmt; ++j) {
              const double expected = tr(j, 1 + k);
              EXPECT_NEAR(st(j, 1 + k), expected, S::rtol * std::max(1.0, std::abs(expected)))
                  << "F = " << f[cmt - 1] << ", cmt " << cmt << ", rate " << rate
                  << ", ss " << ss << ", t = " << r.obs_times[k] << ", compartment " << j + 1;
            }
          }
        }
      }
    }
  }
}

/*
 * A constant steady-state infusion (AMT = 0, II = 0) runs at RATE, not
 * F * RATE, like a long transient infusion of the same rate.
 */
template <typename S>
void check_constant_infusion(const S& solver) {
  const double rate = 150;
  const std::vector<std::vector<double>> biovar{biovar_cases(S::ncmt)[1]};
  // SS = 1, II = 0, AMT = 0 at t = 0
  const std::vector<double> time{0, 0}, ii{0, 0};
  const std::vector<int> evid{1, 0}, cmt{2, 2}, addl{0, 0}, ss{1, 0};
  const Eigen::MatrixXd st = solver(time, std::vector<double>{0, 0},
                                    std::vector<double>{rate, 0}, ii, evid, cmt, addl,
                                    ss, biovar);
  // the same rate from t = -600, long enough to outlast t = 0 for any F
  const std::vector<double> time2{-600, 0};
  const std::vector<int> ss2{0, 0};
  const Eigen::MatrixXd tr = solver(time2, std::vector<double>{2000 * rate, 0},
                                    std::vector<double>{rate, 0}, ii, evid, cmt, addl,
                                    ss2, biovar);
  for (int j = 0; j < S::ncmt; ++j) {
    EXPECT_NEAR(st(j, 1), tr(j, 1), S::rtol * std::max(1.0, std::abs(tr(j, 1))))
        << "compartment " << j + 1;
  }
}

/*
 * Gradients with respect to F, AMT, RATE and II, by reverse mode against
 * central finite differences, for the steady-state and the transient run.
 */
template <typename S>
void check_gradients(const S& solver, double h, double tol) {
  // observations off the dose times and the infusion ends (6.4 h after
  // each dose), where a perturbed event time would cross an observation
  const regimen r{1200, 150, 12, 1, {0.5, 3, 7, 13, 17, 31}};
  const std::vector<double> f = biovar_cases(S::ncmt)[1];
  for (const regimen::schedule& s : {r.steady(1), r.transient()}) {
    // reverse mode, with every F, AMT, RATE and II a parameter
    std::vector<var> amt(s.amt.begin(), s.amt.end()), rate(s.rate.begin(), s.rate.end()),
        ii(s.ii.begin(), s.ii.end());
    std::vector<std::vector<var>> biovar{std::vector<var>(f.begin(), f.end())};
    auto y = solver(s.time, amt, rate, ii, s.evid, s.cmt, s.addl, s.ss, biovar);

    // the inputs that matter: F of the dosing compartment, and the dose's AMT, RATE, II
    struct input {
      std::string name;
      var x;
      std::function<double&(regimen::schedule&, std::vector<double>&)> perturb;
    };
    const int c = r.cmt - 1;
    std::vector<input> inputs{
        {"F", biovar[0][c], [c](regimen::schedule&, std::vector<double>& f) -> double& { return f[c]; }},
        {"AMT", amt[0], [](regimen::schedule& t, std::vector<double>&) -> double& { return t.amt[0]; }},
        {"RATE", rate[0], [](regimen::schedule& t, std::vector<double>&) -> double& { return t.rate[0]; }},
        {"II", ii[0], [](regimen::schedule& t, std::vector<double>&) -> double& { return t.ii[0]; }}};

    // largest |FD| and relative error per input: a vacuous check (an input
    // that does not reach the output) shows up as max |FD| = 0
    std::vector<double> max_fd(inputs.size(), 0), max_err(inputs.size(), 0);
    for (size_t k = 0; k < r.obs_times.size(); ++k) {
      for (int j = 0; j < S::ncmt; ++j) {
        stan::math::set_zero_all_adjoints();
        y(j, 1 + k).grad();
        for (size_t m = 0; m < inputs.size(); ++m) {
          const input& in = inputs[m];
          auto value_at = [&](double delta) {
            regimen::schedule t = s;
            std::vector<double> g = f;
            in.perturb(t, g) += delta;
            return solve(solver, t, std::vector<std::vector<double>>{g})(j, 1 + k);
          };
          const double fd = (value_at(h) - value_at(-h)) / (2 * h);
          EXPECT_NEAR(in.x.adj(), fd, tol * std::max(1.0, std::abs(fd)))
              << in.name << ", ss " << s.ss[0] << ", t = " << r.obs_times[k]
              << ", compartment " << j + 1;
          max_fd[m] = std::max(max_fd[m], std::abs(fd));
          max_err[m] = std::max(max_err[m], std::abs(in.x.adj() - fd) / std::max(1.0, std::abs(fd)));
        }
      }
    }
    for (size_t m = 0; m < inputs.size(); ++m) {
      EXPECT_GT(max_fd[m], 1e-3) << inputs[m].name << ", ss " << s.ss[0];
      std::cout << "  ss " << s.ss[0] << " " << inputs[m].name << ": max |FD| " << max_fd[m]
                << ", max relative error " << max_err[m] << "\n";
    }
    stan::math::recover_memory();
  }
}

}  // namespace

TEST(pmx_ss_biovar_infusion, onecpt_steady_state_is_transient_limit) {
  check_steady_state_is_transient_limit(onecpt_solver());
}

TEST(pmx_ss_biovar_infusion, twocpt_steady_state_is_transient_limit) {
  check_steady_state_is_transient_limit(twocpt_solver());
}

TEST(pmx_ss_biovar_infusion, linode_steady_state_is_transient_limit) {
  check_steady_state_is_transient_limit(linode_solver());
}

TEST(pmx_ss_biovar_infusion, rk45_steady_state_is_transient_limit) {
  check_steady_state_is_transient_limit(rk45_solver());
}

TEST(pmx_ss_biovar_infusion, bdf_steady_state_is_transient_limit) {
  check_steady_state_is_transient_limit(bdf_solver());
}

TEST(pmx_ss_biovar_infusion, constant_infusion_rate_not_scaled) {
  check_constant_infusion(onecpt_solver());
  check_constant_infusion(twocpt_solver());
  check_constant_infusion(rk45_solver());
}

/*
 * Closed forms for the depot of the onecpt model, with F = 0.8, AMT 1200,
 * RATE 150: the infusion runs at 150 for d = F * AMT / RATE = 6.4 h.
 */
TEST(pmx_ss_biovar_infusion, onecpt_depot_closed_form) {
  const double F = 0.8, amt = 1200, rate = 150, ka = 1.2, ii = 12;
  const double d = F * amt / rate;
  const std::vector<std::vector<double>> biovar{{F, 1}};

  // a single infusion at t = 0, observed during and after it
  const std::vector<double> time{0, 3, 6, 10};
  const std::vector<double> a{amt, 0, 0, 0}, r{rate, 0, 0, 0}, i0{0, 0, 0, 0};
  const std::vector<int> evid{1, 0, 0, 0}, cmt{1, 1, 1, 1}, addl{0, 0, 0, 0},
      ss{0, 0, 0, 0};
  const Eigen::MatrixXd y = onecpt_solver()(time, a, r, i0, evid, cmt, addl, ss, biovar);
  for (int k = 1; k < 4; ++k) {
    const double t = time[k];
    const double expected = t <= d ? rate / ka * (1 - std::exp(-ka * t))
                                   : rate / ka * (1 - std::exp(-ka * d)) * std::exp(-ka * (t - d));
    EXPECT_NEAR(y(0, k), expected, 1e-10 * expected) << "t = " << t;
  }

  // the steady state at the dose time
  const double expected = rate / ka * (1 - std::exp(-ka * d)) * std::exp(-ka * (ii - d))
                          / (1 - std::exp(-ka * ii));
  const regimen reg{amt, rate, ii, 1};
  for (const regimen::schedule& s : {reg.steady(1), reg.transient()}) {
    const Eigen::MatrixXd x = solve(onecpt_solver(), s, biovar);
    EXPECT_NEAR(x(0, 1), expected, 1e-9 * expected) << "ss " << s.ss[0];
  }
}

TEST(pmx_ss_biovar_infusion, onecpt_gradients) {
  check_gradients(onecpt_solver(), 1e-5, 1e-6);
}

TEST(pmx_ss_biovar_infusion, rk45_gradients) {
  check_gradients(rk45_solver(), 1e-4, 1e-4);
}
