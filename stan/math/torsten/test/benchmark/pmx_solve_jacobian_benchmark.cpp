/*
 * Value and gradient of pmx_solve_rk45/bdf/adams/dde_rk45 with the
 * Jacobian of the right hand side, (f, jac, nCmt, ...), against nested
 * autodiff, (f, nCmt, ...), on a transit-compartment PK/PD model of
 * N = nt + 4 states.
 *
 * Build like a unit test (no gtest), e.g. with the recipe of the
 * torsten_math primer, at -O3:
 *   g++ -std=c++17 -O3 -DBOOST_DISABLE_ASSERTS -I <math> -isystem ... \
 *     pmx_solve_jacobian_benchmark.cpp <sundials .a> -ltbb -o bench
 * and run ./bench [seconds per measurement].
 *
 * Model: an absorption chain depot -> nt transit compartments -> central,
 * two-compartment disposition, and an indirect response R whose production
 * is inhibited by the concentration C = central / V1:
 *
 *   ODE: R' = kin IC50 / (IC50 + C) - kout R
 *   DDE: R' = kin IC50 / (IC50 + C(t - tau_1)) sqrt(kin / kout / R(t - tau_2))
 *             - kout R
 *
 * theta = (CL, Q, V1, V2, ktr, kin, kout, IC50). Oral doses every 12 h
 * (ADDL 9), observations every 2 h for 5 days; the gradient is that of the
 * sum of all predicted amounts with respect to theta.
 */
#include <stan/math/rev.hpp>
#include <stan/math/torsten/pmx_solve_rk45.hpp>
#include <stan/math/torsten/pmx_solve_bdf.hpp>
#include <stan/math/torsten/pmx_solve_adams.hpp>
#include <stan/math/torsten/pmx_solve_dde_rk45.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using stan::math::var;

struct transit_ir_ode {
  int nt;
  template <typename T0, typename T1, typename T2>
  Eigen::Matrix<stan::return_type_t<T0, T1, T2>, -1, 1>
  operator()(const T0& t, const Eigen::Matrix<T1, -1, 1>& y,
             const std::vector<T2>& theta, const std::vector<double>& x_r,
             const std::vector<int>& x_i, std::ostream* msgs) const {
    using scalar = stan::return_type_t<T0, T1, T2>;
    const T2 &CL = theta[0], &Q = theta[1], &V1 = theta[2], &V2 = theta[3],
             &ktr = theta[4], &kin = theta[5], &kout = theta[6], &IC50 = theta[7];
    const int c = nt + 1, p = nt + 2, r = nt + 3;
    Eigen::Matrix<scalar, -1, 1> dydt(nt + 4);
    dydt(0) = -ktr * y(0);
    for (int i = 1; i <= nt; ++i) {
      dydt(i) = ktr * (y(i - 1) - y(i));
    }
    dydt(c) = ktr * y(nt) - (CL + Q) / V1 * y(c) + Q / V2 * y(p);
    dydt(p) = Q / V1 * y(c) - Q / V2 * y(p);
    dydt(r) = kin * IC50 / (IC50 + y(c) / V1) - kout * y(r);
    return dydt;
  }
};

/**
 * Columns of theta's block: the states' derivatives with respect to the
 * absorption chain and two-compartment disposition, shared by the ODE and
 * the DDE Jacobian. off is the first theta column.
 */
void transit_pk_jacobian(int nt, const Eigen::VectorXd& y,
                         const std::vector<double>& theta, int off,
                         Eigen::MatrixXd& J) {
  const double CL = theta[0], Q = theta[1], V1 = theta[2], V2 = theta[3],
               ktr = theta[4];
  const int c = nt + 1, p = nt + 2;
  for (int i = 0; i <= nt; ++i) {
    J(i, i) = -ktr;
    J(i, off + 4) = -y(i);
    if (i > 0) {
      J(i, i - 1) = ktr;
      J(i, off + 4) += y(i - 1);
    }
  }
  J(c, nt) = ktr;
  J(c, c) = -(CL + Q) / V1;
  J(c, p) = Q / V2;
  J(c, off + 0) = -y(c) / V1;
  J(c, off + 1) = -y(c) / V1 + y(p) / V2;
  J(c, off + 2) = (CL + Q) * y(c) / (V1 * V1);
  J(c, off + 3) = -Q * y(p) / (V2 * V2);
  J(c, off + 4) = y(nt);
  J(p, c) = Q / V1;
  J(p, p) = -Q / V2;
  J(p, off + 1) = y(c) / V1 - y(p) / V2;
  J(p, off + 2) = -Q * y(c) / (V1 * V1);
  J(p, off + 3) = Q * y(p) / (V2 * V2);
}

struct transit_ir_ode_jac {
  int nt;
  Eigen::MatrixXd operator()(double t, const Eigen::VectorXd& y,
                             const std::vector<double>& theta,
                             const std::vector<double>& x_r,
                             const std::vector<int>& x_i,
                             std::ostream* msgs) const {
    const int n = nt + 4, c = nt + 1, r = nt + 3;
    const double V1 = theta[2], kin = theta[5], kout = theta[6], IC50 = theta[7];
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n, n + 8);
    transit_pk_jacobian(nt, y, theta, n, J);
    const double C = y(c) / V1, D = IC50 + C;
    J(r, c) = -kin * IC50 / (D * D) / V1;
    J(r, r) = -kout;
    J(r, n + 2) = kin * IC50 * C / (V1 * D * D);
    J(r, n + 5) = IC50 / D;
    J(r, n + 6) = -y(r);
    J(r, n + 7) = kin * C / (D * D);
    return J;
  }
};

struct transit_ir_dde {
  int nt;
  template <typename T0, typename T1, typename T2, typename T3>
  Eigen::Matrix<stan::return_type_t<T0, T1, T2, T3>, -1, 1>
  operator()(const T0& t, const Eigen::Matrix<T1, -1, 1>& y,
             const Eigen::Matrix<T2, -1, -1>& y_lag,
             const std::vector<T3>& theta, const std::vector<double>& x_r,
             const std::vector<int>& x_i, std::ostream* msgs) const {
    using scalar = stan::return_type_t<T0, T1, T2, T3>;
    const T3 &CL = theta[0], &Q = theta[1], &V1 = theta[2], &V2 = theta[3],
             &ktr = theta[4], &kin = theta[5], &kout = theta[6], &IC50 = theta[7];
    const int c = nt + 1, p = nt + 2, r = nt + 3;
    Eigen::Matrix<scalar, -1, 1> dydt(nt + 4);
    dydt(0) = -ktr * y(0);
    for (int i = 1; i <= nt; ++i) {
      dydt(i) = ktr * (y(i - 1) - y(i));
    }
    dydt(c) = ktr * y(nt) - (CL + Q) / V1 * y(c) + Q / V2 * y(p);
    dydt(p) = Q / V1 * y(c) - Q / V2 * y(p);
    dydt(r) = kin * IC50 / (IC50 + y_lag(c, 0) / V1)
                  * stan::math::sqrt(kin / kout / y_lag(r, 1))
              - kout * y(r);
    return dydt;
  }
};

struct transit_ir_dde_jac {
  int nt;
  Eigen::MatrixXd operator()(double t, const Eigen::VectorXd& y,
                             const Eigen::MatrixXd& y_lag,
                             const std::vector<double>& theta,
                             const std::vector<double>& x_r,
                             const std::vector<int>& x_i,
                             std::ostream* msgs) const {
    const int n = nt + 4, c = nt + 1, r = nt + 3, off = 3 * n;
    const double V1 = theta[2], kin = theta[5], kout = theta[6], IC50 = theta[7];
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n, off + 8);
    transit_pk_jacobian(nt, y, theta, off, J);
    const double C = y_lag(c, 0) / V1, D = IC50 + C, R_lag = y_lag(r, 1);
    const double fb = std::sqrt(kin / kout / R_lag);
    const double inh = IC50 / D;
    J(r, r) = -kout;
    J(r, n + c) = -kin * IC50 / (D * D) / V1 * fb;
    J(r, 2 * n + r) = -kin * inh * fb / (2 * R_lag);
    J(r, off + 2) = kin * IC50 * C / (V1 * D * D) * fb;
    J(r, off + 5) = 1.5 * inh * fb;
    J(r, off + 6) = -kin * inh * fb / (2 * kout) - y(r);
    J(r, off + 7) = kin * C / (D * D) * fb;
    return J;
  }
};

struct schedule {
  std::vector<double> time, amt, rate, ii;
  std::vector<int> evid, cmt, addl, ss;
  schedule() {
    add(0, 1, 1, 100, 12, 9);
    for (double t = 2; t <= 120; t += 2) {
      add(t, 0, 1, 0, 0, 0);
    }
  }
  void add(double t, int ev, int c, double a, double i, int ad) {
    time.push_back(t);
    evid.push_back(ev);
    cmt.push_back(c);
    amt.push_back(a);
    rate.push_back(0);
    ii.push_back(i);
    addl.push_back(ad);
    ss.push_back(0);
  }
};

const std::vector<double> theta_ref{5.0, 8.0, 35.0, 105.0, 2.0, 4.0, 0.4, 0.6};

/**
 * Value and gradient of the sum of all predicted amounts.
 */
template <typename Solve>
double value_and_gradient(const Solve& solve, std::vector<double>& grad) {
  stan::math::nested_rev_autodiff nested;
  std::vector<var> theta(theta_ref.begin(), theta_ref.end());
  var total = solve(theta).sum();
  total.grad();
  grad.resize(theta.size());
  for (size_t k = 0; k < theta.size(); ++k) {
    grad[k] = theta[k].adj();
  }
  return total.val();
}

/**
 * Median over 5 rounds of the time per value-and-gradient, each round
 * repeating until it takes at least the given seconds / 5.
 */
template <typename Solve>
double seconds_per_gradient(const Solve& solve, double seconds) {
  std::vector<double> grad;
  value_and_gradient(solve, grad);  // warm-up
  std::vector<double> rounds;
  for (int round = 0; round < 5; ++round) {
    int reps = 0;
    const auto start = std::chrono::steady_clock::now();
    double elapsed = 0;
    do {
      value_and_gradient(solve, grad);
      ++reps;
      elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                    .count();
    } while (elapsed < seconds / 5);
    rounds.push_back(elapsed / reps);
  }
  std::sort(rounds.begin(), rounds.end());
  return rounds[2];
}

template <typename SolveAD, typename SolveJac>
void compare(const std::string& name, int n, const SolveAD& solve_ad,
             const SolveJac& solve_jac, double seconds) {
  std::vector<double> grad_ad, grad_jac;
  const double val_ad = value_and_gradient(solve_ad, grad_ad);
  const double val_jac = value_and_gradient(solve_jac, grad_jac);
  double diff = std::abs(val_jac - val_ad) / (1 + std::abs(val_ad));
  for (size_t k = 0; k < grad_ad.size(); ++k) {
    diff = std::max(diff, std::abs(grad_jac[k] - grad_ad[k]) / (1 + std::abs(grad_ad[k])));
  }
  const double t_ad = seconds_per_gradient(solve_ad, seconds);
  const double t_jac = seconds_per_gradient(solve_jac, seconds);
  std::printf("%-18s N=%3d  AD %10.3f ms  jac %10.3f ms  speedup %6.2fx  max rel diff %.1e\n",
              name.c_str(), n, 1e3 * t_ad, 1e3 * t_jac, t_ad / t_jac, diff);
  std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
  const double seconds = argc > 1 ? std::atof(argv[1]) : 2.0;
  const schedule s;
  const double tol = 1e-8;
  const int max_steps = 100000000;
  for (int nt : {0, 8, 20, 40}) {
    const int n = nt + 4;
    const transit_ir_ode f{nt};
    const transit_ir_ode_jac jac{nt};
    auto ode = [&](auto solver) {
      return [&, solver](const auto&... jac_) {
        return [&, solver, jac_...](const std::vector<var>& theta) {
          return solver(f, jac_..., n, s.time, s.amt, s.rate, s.ii, s.evid, s.cmt,
                        s.addl, s.ss, theta, tol, tol, max_steps, nullptr);
        };
      };
    };
    auto rk45 = ode([](const auto&... a) { return torsten::pmx_solve_rk45(a...); });
    auto bdf = ode([](const auto&... a) { return torsten::pmx_solve_bdf(a...); });
    auto adams = ode([](const auto&... a) { return torsten::pmx_solve_adams(a...); });
    compare("pmx_solve_rk45", n, rk45(), rk45(jac), seconds);
    compare("pmx_solve_bdf", n, bdf(), bdf(jac), seconds);
    compare("pmx_solve_adams", n, adams(), adams(jac), seconds);

    const transit_ir_dde g{nt};
    const transit_ir_dde_jac g_jac{nt};
    Eigen::VectorXd tau(2);
    tau << 1.5, 4.0;
    Eigen::VectorXd y_hist = Eigen::VectorXd::Zero(n);
    y_hist(n - 1) = 10.0;  // R at baseline kin / kout
    auto dde = [&](const auto&... jac_) {
      return [&, jac_...](const std::vector<var>& theta) {
        return torsten::pmx_solve_dde_rk45(g, jac_..., n, s.time, s.amt, s.rate,
                                           s.ii, s.evid, s.cmt, s.addl, s.ss,
                                           theta, tau, y_hist, tol, tol,
                                           max_steps, nullptr);
      };
    };
    compare("pmx_solve_dde_rk45", n, dde(), dde(g_jac), seconds);
  }
  return 0;
}
