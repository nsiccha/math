#ifndef STAN_MATH_PRIM_FUNCTOR_DDE_RETARD_HPP
#define STAN_MATH_PRIM_FUNCTOR_DDE_RETARD_HPP

// The integrator in this file is a C++ adaptation of RETARD (version of
// September 30, 1995) by E. Hairer and G. Wanner, described in Section II.17
// of E. Hairer, S.P. Norsett and G. Wanner, "Solving Ordinary Differential
// Equations I. Nonstiff Problems", 2nd edition, Springer (1993). The original
// Fortran code is distributed at https://www.unige.ch/~hairer/software.html
// under the following licence
// (https://www.unige.ch/~hairer/prog/licence.txt):
//
//   Copyright (c) 2004, UNIGE
//
//   Redistribution and use in source and binary forms, with or without
//   modification, are permitted provided that the following conditions are
//   met:
//
//   - Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
//   - Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
//   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
//   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
//   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR
//   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// The full licence text is also in licenses/retard-license.txt.
//
// Differences from the Fortran original:
//  - all solver state lives in local objects of one call (no COMMON blocks or
//    other static state), so concurrent solves on different threads are safe;
//  - delays are constant, the initial history is the constant initial state,
//    and lagged values are supplied to the right hand side by the integrator
//    (instead of the right hand side calling YLAG);
//  - the derivative of the dense output is available for lagged arguments
//    (needed for sensitivities with respect to the delays);
//  - the breakpoints t0 + sum_j k_j tau_j are generated internally (see
//    dde_breakpoints) instead of being passed in as WORK(21), ...;
//  - output at the requested times is computed from the dense output;
//  - the stiffness detection is not ported;
//  - failures are reported through Stan's error functions.

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace stan {
namespace math {
namespace internal {

/**
 * Breakpoints of a constant-delay DDE whose initial history is constant.
 *
 * The derivative of the solution jumps at t0 (the history is constant, the
 * solution is not), and this discontinuity propagates to
 * t0 + sum_j k_j tau_j (k_j >= 0 integers), where the jump moves to higher
 * derivatives as sum_j k_j grows. The integrator steps exactly onto
 *  - every multiple t0 + k tau_j (k = 1, 2, ...) of every delay, and
 *  - every mixed combination t0 + sum_j k_j tau_j with at least two nonzero
 *    k_j and total order sum_j k_j <= max_mixed_order (= 5),
 * up to the final output time. The sensitivities with respect to a delay have
 * a jump in their first derivative at t0 + tau_j, so a breakpoint of total
 * order k leaves at most a jump in the k-th derivative there; beyond order 5
 * these are invisible to the fifth order Dormand-Prince method.
 *
 * The multiples of each delay are generated lazily, so their number never
 * needs to be stored. Breakpoints closer than a relative tolerance are merged
 * (the first one is kept), and a breakpoint closer than that tolerance to the
 * final time is replaced by the final time.
 */
class dde_breakpoints {
 public:
  static constexpr int max_mixed_order = 5;

  /**
   * @param t0 initial time
   * @param tau delays (all positive)
   * @param t_end final time (> t0)
   */
  dde_breakpoints(double t0, const std::vector<double>& tau, double t_end)
      : t0_(t0),
        tau_(tau),
        t_end_(t_end),
        tol_(100 * std::numeric_limits<double>::epsilon()
             * std::max(std::fabs(t0), std::fabs(t_end))),
        mixed_pos_(0) {
    if (tau_.size() > 1) {
      add_mixed(0, 0, 0.0, 0);
      std::sort(mixed_.begin(), mixed_.end());
    }
  }

  /**
   * Tolerance below which two breakpoints are merged.
   */
  double tolerance() const { return tol_; }

  /**
   * The first breakpoint after x (at least tolerance() after x), or the final
   * time if there is none before it.
   *
   * @param x current time; must be less than the final time by more than
   *   tolerance()
   */
  double next(double x) {
    const double lower = x + tol_;
    double g = t_end_;
    for (double tau_j : tau_) {
      double k = std::floor((lower - t0_) / tau_j) + 1;
      double v = t0_ + k * tau_j;
      while (v <= lower) {
        k += 1;
        v = t0_ + k * tau_j;
      }
      g = std::min(g, v);
    }
    while (mixed_pos_ < mixed_.size() && mixed_[mixed_pos_] <= lower) {
      ++mixed_pos_;
    }
    if (mixed_pos_ < mixed_.size()) {
      g = std::min(g, mixed_[mixed_pos_]);
    }
    if (t_end_ - g <= tol_) {
      g = t_end_;
    }
    return g;
  }

 private:
  void add_mixed(size_t j, int order, double sum, int nonzero) {
    if (j == tau_.size()) {
      if (nonzero >= 2) {
        mixed_.push_back(t0_ + sum);
      }
      return;
    }
    for (int k = 0; order + k <= max_mixed_order; ++k) {
      const double s = sum + k * tau_[j];
      if (t0_ + s > t_end_ + tol_) {
        break;
      }
      add_mixed(j + 1, order + k, s, nonzero + (k > 0));
    }
  }

  const double t0_;
  const std::vector<double>& tau_;
  const double t_end_;
  const double tol_;
  std::vector<double> mixed_;
  size_t mixed_pos_;
};

/**
 * History of a DDE solution: the dense output coefficients of Shampine
 * (RETARD's CONT array) of every accepted step that can still be reached by a
 * lagged argument.
 *
 * On the step [x, x + h], with theta = (s - x) / h, the dense output is
 *   c1 + theta (c2 + (1 - theta) (c3 + theta (c4 + (1 - theta) c5))).
 */
class dde_dense_history {
 public:
  explicit dde_dense_history(size_t n) : n_(n), first_(0) {}

  bool empty() const { return first_ == x_.size(); }

  /**
   * Store the dense output of the step [x, x + h].
   */
  void push(double x, double h, const Eigen::VectorXd& y,
            const Eigen::VectorXd& y_new, const Eigen::VectorXd& k1,
            const Eigen::VectorXd& k7, const Eigen::VectorXd& c5) {
    const size_t offset = cont_.size();
    cont_.resize(offset + 5 * n_);
    Eigen::Map<Eigen::MatrixXd> c(cont_.data() + offset, n_, 5);
    c.col(0) = y;
    c.col(1) = y_new - y;
    c.col(2) = h * k1 - c.col(1);
    c.col(3) = c.col(1) - h * k7 - c.col(2);
    c.col(4) = c5;
    x_.push_back(x);
    h_.push_back(h);
  }

  /**
   * Index of the retained step whose interval contains s. Arguments outside
   * the retained steps (by rounding) map to the first or last step.
   */
  size_t locate(double s) const {
    auto begin = x_.begin() + first_;
    auto it = std::upper_bound(begin, x_.end(), s);
    if (it == begin) {
      return first_;
    }
    return static_cast<size_t>(it - x_.begin()) - 1;
  }

  /**
   * Dense output of all components at s, using step k.
   */
  template <typename Out>
  void value(size_t k, double s, Out&& out) const {
    const double theta = (s - x_[k]) / h_[k];
    const double theta1 = 1.0 - theta;
    const auto c = coefficients(k);
    out = c.col(0)
          + theta
                * (c.col(1)
                   + theta1
                         * (c.col(2) + theta * (c.col(3) + theta1 * c.col(4))));
  }

  /**
   * Time derivative of the dense output of the first m components at s,
   * using step k.
   */
  template <typename Out>
  void derivative(size_t k, double s, size_t m, Out&& out) const {
    const double theta = (s - x_[k]) / h_[k];
    const double theta1 = 1.0 - theta;
    const auto c = coefficients(k).topRows(m);
    const Eigen::VectorXd a = c.col(3) + theta1 * c.col(4);
    const Eigen::VectorXd b = c.col(2) + theta * a;
    const Eigen::VectorXd db = a - theta * c.col(4);
    const Eigen::VectorXd dc = theta1 * db - b;
    out = (c.col(1) + theta1 * b + theta * dc) / h_[k];
  }

  /**
   * Forget the steps that end before s, always keeping the last step.
   */
  void forget_before(double s) {
    while (x_.size() - first_ > 1 && x_[first_] + h_[first_] < s) {
      ++first_;
    }
    // compact the storage once the forgotten steps dominate it
    if (first_ > 64 && 2 * first_ > x_.size()) {
      x_.erase(x_.begin(), x_.begin() + first_);
      h_.erase(h_.begin(), h_.begin() + first_);
      cont_.erase(cont_.begin(), cont_.begin() + 5 * n_ * first_);
      first_ = 0;
    }
  }

 private:
  Eigen::Map<const Eigen::MatrixXd> coefficients(size_t k) const {
    return Eigen::Map<const Eigen::MatrixXd>(cont_.data() + 5 * n_ * k, n_, 5);
  }

  const size_t n_;
  size_t first_;
  std::vector<double> x_;
  std::vector<double> h_;
  std::vector<double> cont_;
};

/**
 * Integrate the constant-delay DDE system
 *   z'(t) = F(t, z(t), z(t - tau_1), ..., z(t - tau_D)),
 *   z(t) = z0 for t <= t0,
 * with the explicit Runge-Kutta method of Dormand and Prince of order 5(4),
 * step size control, Shampine's dense output for the history, and steps
 * landing exactly on the breakpoints of dde_breakpoints (a C++ adaptation of
 * RETARD by Hairer and Wanner; see the notice at the top of this file).
 *
 * As in RETARD, the step size never exceeds the smallest delay, so lagged
 * arguments always lie in the already computed past (or the initial
 * history). Until the step that lands on the breakpoint t0 + tau_j, the
 * lagged argument t - tau_j lies in the initial history: its value is z0 and
 * its derivative is zero. From that breakpoint on the dense output is used.
 * The first derivative of the solution jumps at t0, so the stage derivative
 * at the start of the following step is recomputed instead of reusing the
 * last stage of the step that ended there (RETARD's IRTRN = 3 / 4 logic) if
 * the system uses derivatives of lagged arguments.
 *
 * @tparam System Type of the system. It must provide
 *   - <code>size_t size() const</code>, the number of components;
 *   - <code>size_t num_lag_derivatives() const</code>, the number m of
 *     leading components whose lagged derivatives the system needs;
 *   - <code>void operator()(double t, const Eigen::VectorXd& z,
 *       const Eigen::MatrixXd& z_lag, const Eigen::MatrixXd& z_lag_deriv,
 *       Eigen::VectorXd& dz_dt)</code>, where column j of the size() x D
 *     matrix z_lag is z(t - tau_j) and column j of the m x D matrix
 *     z_lag_deriv is the time derivative of the first m components at
 *     t - tau_j.
 * @tparam Observer Type of the observer, called as
 *   <code>observer(i, z)</code> with the state z at ts[i] for each i in order.
 *
 * @param function_name name of the calling function, for error messages
 * @param system right hand side of the system
 * @param z0 initial state (and constant initial history)
 * @param t0 initial time
 * @param ts output times, strictly increasing and greater than t0
 * @param tau positive delays
 * @param relative_tolerance relative tolerance of the local error
 * @param absolute_tolerance absolute tolerance of the local error
 * @param max_num_steps maximum number of steps (accepted or rejected) between
 *   two output times
 * @param observer observer of the solution at the output times
 * @throw std::domain_error if more than max_num_steps steps are needed to
 *   reach an output time, or if the step size becomes too small (which is
 *   the case when the right hand side keeps returning non-finite values)
 */
template <typename System, typename Observer>
inline void dde_retard_integrate(const char* function_name, System& system,
                                 const Eigen::VectorXd& z0, double t0,
                                 const std::vector<double>& ts,
                                 const std::vector<double>& tau,
                                 double relative_tolerance,
                                 double absolute_tolerance,
                                 long int max_num_steps,  // NOLINT(runtime/int)
                                 Observer&& observer) {
  // Runge-Kutta coefficients of Dormand and Prince (1980), RETARD's CDOPRI
  static constexpr double c2 = 0.2;
  static constexpr double c3 = 0.3;
  static constexpr double c4 = 0.8;
  static constexpr double c5 = 8.0 / 9.0;
  static constexpr double a21 = 0.2;
  static constexpr double a31 = 3.0 / 40.0;
  static constexpr double a32 = 9.0 / 40.0;
  static constexpr double a41 = 44.0 / 45.0;
  static constexpr double a42 = -56.0 / 15.0;
  static constexpr double a43 = 32.0 / 9.0;
  static constexpr double a51 = 19372.0 / 6561.0;
  static constexpr double a52 = -25360.0 / 2187.0;
  static constexpr double a53 = 64448.0 / 6561.0;
  static constexpr double a54 = -212.0 / 729.0;
  static constexpr double a61 = 9017.0 / 3168.0;
  static constexpr double a62 = -355.0 / 33.0;
  static constexpr double a63 = 46732.0 / 5247.0;
  static constexpr double a64 = 49.0 / 176.0;
  static constexpr double a65 = -5103.0 / 18656.0;
  static constexpr double a71 = 35.0 / 384.0;
  static constexpr double a73 = 500.0 / 1113.0;
  static constexpr double a74 = 125.0 / 192.0;
  static constexpr double a75 = -2187.0 / 6784.0;
  static constexpr double a76 = 11.0 / 84.0;
  static constexpr double e1 = 71.0 / 57600.0;
  static constexpr double e3 = -71.0 / 16695.0;
  static constexpr double e4 = 71.0 / 1920.0;
  static constexpr double e5 = -17253.0 / 339200.0;
  static constexpr double e6 = 22.0 / 525.0;
  static constexpr double e7 = -1.0 / 40.0;
  // dense output of Shampine (1986)
  static constexpr double d1 = -12715105075.0 / 11282082432.0;
  static constexpr double d3 = 87487479700.0 / 32700410799.0;
  static constexpr double d4 = -10690763975.0 / 1880347072.0;
  static constexpr double d5 = 701980252875.0 / 199316789632.0;
  static constexpr double d6 = -1453857185.0 / 822651844.0;
  static constexpr double d7 = 69997945.0 / 29380423.0;
  // step size control parameters (RETARD defaults)
  static constexpr double uround = 10 * 2.3e-16;
  static constexpr double safe = 0.9;
  static constexpr double facc1 = 1.0 / 0.2;
  static constexpr double facc2 = 1.0 / 10.0;
  static constexpr double beta = 0.04;
  static constexpr double expo1 = 0.2 - beta * 0.75;

  const size_t n = z0.size();
  const size_t m = system.num_lag_derivatives();
  const size_t num_delays = tau.size();
  const double t_end = ts.back();
  const double rtol = relative_tolerance;
  const double atol = absolute_tolerance;

  dde_breakpoints breakpoints(t0, tau, t_end);
  dde_dense_history history(n);

  // the maximal step size is the smallest delay (RETARD sets HMAX this way
  // during the first evaluation of the right hand side)
  double hmax = t_end - t0;
  double tau_min = std::numeric_limits<double>::infinity();
  double tau_max = 0;
  for (double tau_j : tau) {
    tau_min = std::min(tau_min, tau_j);
    tau_max = std::max(tau_max, tau_j);
  }
  hmax = std::min(hmax, tau_min);
  if (!(tau_min > breakpoints.tolerance())) {
    throw_domain_error(function_name, "smallest delay", tau_min, "is ",
                       ", but must be larger than 100 machine epsilons "
                       "times the largest absolute integration time");
  }

  // steps starting before history_until[j] see z(t - tau_j) in the initial
  // history; it is set to the breakpoint landed on for t0 + tau_j
  std::vector<double> history_until(num_delays,
                                    std::numeric_limits<double>::infinity());

  Eigen::MatrixXd z_lag(n, num_delays);
  Eigen::MatrixXd z_lag_deriv(m, num_delays);
  auto rhs = [&](double x_step, double t, const Eigen::VectorXd& z,
                 Eigen::VectorXd& dz_dt) {
    for (size_t j = 0; j < num_delays; ++j) {
      if (x_step < history_until[j]) {
        z_lag.col(j) = z0;
        z_lag_deriv.col(j).setZero();
      } else {
        const double s = t - tau[j];
        const size_t k = history.locate(s);
        history.value(k, s, z_lag.col(j));
        if (m > 0) {
          history.derivative(k, s, m, z_lag_deriv.col(j));
        }
      }
    }
    system(t, z, z_lag, z_lag_deriv, dz_dt);
  };

  auto scale = [&](double y_i, double y1_i) {
    return atol + rtol * std::max(std::fabs(y_i), std::fabs(y1_i));
  };

  Eigen::VectorXd y = z0;
  Eigen::VectorXd y1(n);
  Eigen::VectorXd ysti(n);
  Eigen::VectorXd k1(n);
  Eigen::VectorXd k2(n);
  Eigen::VectorXd k3(n);
  Eigen::VectorXd k4(n);
  Eigen::VectorXd k5(n);
  Eigen::VectorXd k6(n);
  Eigen::VectorXd k7(n);
  Eigen::VectorXd c5_dense(n);
  Eigen::VectorXd err_vec(n);

  double x = t0;
  rhs(x, x, y, k1);

  // initial step size (RETARD's HINIT, order 5)
  double h;
  {
    double dnf = 0;
    double dny = 0;
    for (size_t i = 0; i < n; ++i) {
      const double sk = atol + rtol * std::fabs(y[i]);
      dnf += (k1[i] / sk) * (k1[i] / sk);
      dny += (y[i] / sk) * (y[i] / sk);
    }
    if (dnf <= 1e-10 || dny <= 1e-10) {
      h = 1e-6;
    } else {
      h = std::sqrt(dny / dnf) * 0.01;
    }
    h = std::min(h, hmax);
    y1 = y + h * k1;
    rhs(x, x + h, y1, k2);
    double der2 = 0;
    for (size_t i = 0; i < n; ++i) {
      const double sk = atol + rtol * std::fabs(y[i]);
      der2 += ((k2[i] - k1[i]) / sk) * ((k2[i] - k1[i]) / sk);
    }
    der2 = std::sqrt(der2) / h;
    const double der12 = std::max(std::fabs(der2), std::sqrt(dnf));
    double h1;
    if (der12 <= 1e-15) {
      h1 = std::max(1e-6, std::fabs(h) * 1e-3);
    } else {
      h1 = std::pow(0.01 / der12, 1.0 / 5.0);
    }
    h = std::min({100 * h, h1, hmax});
  }

  double facold = 1e-4;
  bool reject = false;
  bool last = false;
  double xend = breakpoints.next(x);
  size_t next_output = 0;
  long int num_steps = 0;  // NOLINT(runtime/int)

  while (true) {
    if (num_steps >= max_num_steps) {
      throw_domain_error(function_name, "", ts[next_output],
                         "Failed to integrate to next output time (",
                         ") in less than max_num_steps steps");
    }
    if (!(std::fabs(h) > std::fabs(x) * uround)) {
      throw_domain_error(function_name, "", x,
                         "Failed to integrate: the step size became too "
                         "small at time (",
                         "); the right hand side is likely not finite");
    }
    if (x + 1.01 * h - xend > 0) {
      h = xend - x;
      last = true;
    } else if (x + 1.8 * h - xend > 0) {
      h = (xend - x) * 0.55;
    }
    const double xph = last ? xend : x + h;
    ++num_steps;

    // the stages; k1 is the derivative at (x, y)
    y1 = y + h * a21 * k1;
    rhs(x, x + c2 * h, y1, k2);
    y1 = y + h * (a31 * k1 + a32 * k2);
    rhs(x, x + c3 * h, y1, k3);
    y1 = y + h * (a41 * k1 + a42 * k2 + a43 * k3);
    rhs(x, x + c4 * h, y1, k4);
    y1 = y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4);
    rhs(x, x + c5 * h, y1, k5);
    ysti = y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5);
    rhs(x, xph, ysti, k6);
    y1 = y + h * (a71 * k1 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6);
    rhs(x, xph, y1, k7);
    c5_dense = h * (d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6 + d7 * k7);
    err_vec = h * (e1 * k1 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6 + e7 * k7);

    // error estimation
    double err = 0;
    for (size_t i = 0; i < n; ++i) {
      const double r = err_vec[i] / scale(y[i], y1[i]);
      err += r * r;
    }
    err = std::sqrt(err / n);
    // a non-finite error estimate or state rejects the step and shrinks it
    // as much as allowed
    const bool finite = std::isfinite(err) && y1.allFinite();

    // computation of hnew, with Lund stabilization, subject to
    // 0.2 <= hnew / h <= 10
    const double fac11 = std::pow(err, expo1);
    double hnew
        = h
          / std::max(facc2,
                     std::min(facc1, fac11 / std::pow(facold, beta) / safe));
    if (finite && err <= 1.0) {
      // step is accepted
      facold = std::max(err, 1.0e-4);
      history.push(x, h, y, y1, k1, k7, c5_dense);
      while (next_output < ts.size() && ts[next_output] <= xph) {
        if (ts[next_output] == xph) {
          observer(next_output, y1);
        } else {
          Eigen::VectorXd z_out(n);
          const double s = ts[next_output];
          history.value(history.locate(s), s, z_out);
          observer(next_output, z_out);
        }
        ++next_output;
        num_steps = 0;
      }
      k1 = k7;
      y = y1;
      x = xph;
      if (last) {
        if (x == t_end) {
          return;
        }
        // lagged arguments t - tau_j leave the initial history at the
        // breakpoint landed on for t0 + tau_j; the derivative of the
        // solution jumps at t0, so recompute the first stage derivative
        bool recompute = false;
        for (size_t j = 0; j < num_delays; ++j) {
          if (history_until[j] == std::numeric_limits<double>::infinity()
              && t0 + tau[j] <= x + breakpoints.tolerance()) {
            history_until[j] = x;
            recompute = true;
          }
        }
        if (recompute && m > 0) {
          rhs(x, x, y, k1);
        }
        last = false;
        xend = breakpoints.next(x);
        hnew = 0.9 * hnew;
      }
      history.forget_before(x - tau_max);
      if (std::fabs(hnew) > hmax) {
        hnew = hmax;
      }
      if (reject) {
        hnew = std::min(std::fabs(hnew), std::fabs(h));
      }
      reject = false;
    } else {
      // step is rejected
      hnew = finite ? h / std::min(facc1, fac11 / safe) : h / facc1;
      reject = true;
      last = false;
    }
    h = hnew;
  }
}

}  // namespace internal
}  // namespace math
}  // namespace stan
#endif
