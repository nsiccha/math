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
//  - all solver state lives in one integrator object (no COMMON blocks or
//    other static state), so concurrent solves on different threads are safe,
//    and the integration can be stopped and resumed;
//  - delays are constant, the initial history is a constant vector, and
//    lagged values are supplied to the right hand side by the integrator
//    (instead of the right hand side calling YLAG);
//  - the state may jump and the right hand side may change at the times the
//    integration is stopped at, with the jump conditions of the forward
//    sensitivities applied by the integrator;
//  - the derivative of the dense output is available for lagged arguments
//    (needed for sensitivities with respect to the delays);
//  - the breakpoints are generated internally from the initial time, the
//    jumps and the changes of the right hand side (see dde_breakpoints)
//    instead of being passed in as WORK(21), ...;
//  - output at the requested times is computed from the dense output;
//  - the stiffness detection is not ported;
//  - failures are reported through Stan's error functions.

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <vector>

namespace stan {
namespace math {
namespace internal {

/**
 * Breakpoints of a constant-delay DDE: the times at which derivatives of the
 * solution can jump, onto which the integrator steps exactly.
 *
 * A discontinuity of the solution or of one of its derivatives at a time u
 * propagates to u + sum_j k_j tau_j (k_j >= 0 integers), where it moves to
 * higher derivatives as the total order sum_j k_j grows. There are two kinds
 * of sources:
 *  - the initial time t0, where the derivative of the solution jumps (the
 *    history is constant, the solution is not). From it the breakpoints are
 *    every multiple t0 + k tau_j (k = 1, 2, ...) of every delay, and every
 *    mixed combination t0 + sum_j k_j tau_j with at least two nonzero k_j and
 *    total order sum_j k_j <= max_mixed_order (= 5);
 *  - a later time u at which the state jumps or the right hand side changes.
 *    From it the breakpoints are u + sum_j k_j tau_j with total order
 *    1 <= sum_j k_j <= max_mixed_order.
 * The sensitivities with respect to a delay have a jump in their first
 * derivative at t0 + tau_j, so a breakpoint of total order k leaves at most a
 * jump in the k-th derivative there; beyond order 5 these are invisible to
 * the fifth order Dormand-Prince method.
 *
 * The multiples of each delay from t0 are generated lazily, so their number
 * never needs to be stored. Breakpoints closer than a tolerance are merged
 * (the first one is kept), and a breakpoint closer than that tolerance to the
 * end of the integration is replaced by it.
 */
class dde_breakpoints {
 public:
  static constexpr int max_mixed_order = 5;

  /**
   * @param tau delays (all positive)
   */
  explicit dde_breakpoints(const std::vector<double>& tau)
      : tau_(tau), has_initial_(false), t0_(0) {}

  /**
   * Forget all sources and set the initial time.
   *
   * @param t0 initial time
   */
  void restart(double t0) {
    points_ = min_heap();
    has_initial_ = true;
    t0_ = t0;
    if (tau_.size() > 1) {
      add_mixed(0, 0, 0.0, 0);
    }
  }

  /**
   * Add the breakpoints u + sum_j k_j tau_j, 1 <= sum_j k_j <= 5, of a jump of
   * the state or a change of the right hand side at time u.
   *
   * @param u time of the jump or change
   */
  void add_source(double u) { add_propagated(u, 0, 0, 0.0); }

  /**
   * The first breakpoint after x (at least tol after x), or t_end if there is
   * none before it or if it is within tol of t_end.
   *
   * @param x current time; must be less than t_end by more than tol
   * @param t_end end of the integration
   * @param tol tolerance below which breakpoints are merged
   */
  double next(double x, double t_end, double tol) {
    const double lower = x + tol;
    double g = t_end;
    if (has_initial_) {
      for (double tau_j : tau_) {
        double k = std::floor((lower - t0_) / tau_j) + 1;
        double v = t0_ + k * tau_j;
        while (v <= lower) {
          k += 1;
          v = t0_ + k * tau_j;
        }
        g = std::min(g, v);
      }
    }
    while (!points_.empty() && points_.top() <= lower) {
      points_.pop();
    }
    if (!points_.empty()) {
      g = std::min(g, points_.top());
    }
    if (t_end - g <= tol) {
      g = t_end;
    }
    return g;
  }

 private:
  using min_heap
      = std::priority_queue<double, std::vector<double>, std::greater<double>>;

  void add_mixed(size_t j, int order, double sum, int nonzero) {
    if (j == tau_.size()) {
      if (nonzero >= 2) {
        points_.push(t0_ + sum);
      }
      return;
    }
    for (int k = 0; order + k <= max_mixed_order; ++k) {
      add_mixed(j + 1, order + k, sum + k * tau_[j], nonzero + (k > 0));
    }
  }

  void add_propagated(double u, size_t j, int order, double sum) {
    if (j == tau_.size()) {
      if (order > 0) {
        points_.push(u + sum);
      }
      return;
    }
    for (int k = 0; order + k <= max_mixed_order; ++k) {
      add_propagated(u, j + 1, order + k, sum + k * tau_[j]);
    }
  }

  const std::vector<double>& tau_;
  bool has_initial_;
  double t0_;
  min_heap points_;
};

/**
 * History of a DDE solution: the dense output coefficients of Shampine
 * (RETARD's CONT array) of every accepted step that can still be reached by a
 * lagged argument.
 *
 * On the step [x, x + h], with theta = (s - x) / h, the dense output is
 *   c1 + theta (c2 + (1 - theta) (c3 + theta (c4 + (1 - theta) c5))).
 *
 * Steps are addressed by a global index, the number of steps pushed before
 * them since the last clear(), which stays valid when old steps are
 * forgotten.
 */
class dde_dense_history {
 public:
  explicit dde_dense_history(size_t n) : n_(n), offset_(0), first_(0) {}

  /**
   * Forget all steps.
   */
  void clear() {
    x_.clear();
    h_.clear();
    cont_.clear();
    offset_ = 0;
    first_ = 0;
  }

  /**
   * Global index of the next step to be pushed.
   */
  size_t end() const { return offset_ + x_.size(); }

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
   * Global index of the retained step among the steps [begin, end) whose
   * interval contains s. Arguments outside these steps (by rounding) map to
   * the first or last of them.
   *
   * @param s time
   * @param begin global index of the first step to consider
   * @param end global index one past the last step to consider; there must be
   *   a retained step in [begin, end)
   */
  size_t locate(double s, size_t begin, size_t end) const {
    auto first = x_.begin() + (std::max(begin, first_) - offset_);
    auto last = x_.begin() + (end - offset_);
    auto it = std::upper_bound(first, last, s);
    if (it == first) {
      return static_cast<size_t>(first - x_.begin()) + offset_;
    }
    return static_cast<size_t>(it - x_.begin()) - 1 + offset_;
  }

  /**
   * Dense output of all components at s, using step k.
   */
  template <typename Out>
  void value(size_t k, double s, Out&& out) const {
    const size_t i = k - offset_;
    const double theta = (s - x_[i]) / h_[i];
    const double theta1 = 1.0 - theta;
    const auto c = coefficients(i);
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
    const size_t i = k - offset_;
    const double theta = (s - x_[i]) / h_[i];
    const double theta1 = 1.0 - theta;
    const auto c = coefficients(i).topRows(m);
    const Eigen::VectorXd a = c.col(3) + theta1 * c.col(4);
    const Eigen::VectorXd b = c.col(2) + theta * a;
    const Eigen::VectorXd db = a - theta * c.col(4);
    const Eigen::VectorXd dc = theta1 * db - b;
    out = (c.col(1) + theta1 * b + theta * dc) / h_[i];
  }

  /**
   * Forget the steps that end before s, always keeping the last step.
   */
  void forget_before(double s) {
    while (end() - first_ > 1
           && x_[first_ - offset_] + h_[first_ - offset_] < s) {
      ++first_;
    }
    // compact the storage once the forgotten steps dominate it
    const size_t forgotten = first_ - offset_;
    if (forgotten > 64 && 2 * forgotten > x_.size()) {
      x_.erase(x_.begin(), x_.begin() + forgotten);
      h_.erase(h_.begin(), h_.begin() + forgotten);
      cont_.erase(cont_.begin(), cont_.begin() + 5 * n_ * forgotten);
      offset_ = first_;
    }
  }

 private:
  Eigen::Map<const Eigen::MatrixXd> coefficients(size_t i) const {
    return Eigen::Map<const Eigen::MatrixXd>(cont_.data() + 5 * n_ * i, n_, 5);
  }

  const size_t n_;
  size_t offset_;
  size_t first_;
  std::vector<double> x_;
  std::vector<double> h_;
  std::vector<double> cont_;
};

/**
 * Resumable integrator of the constant-delay DDE system
 *   z'(t) = F(t, z(t), z(t - tau_1), ..., z(t - tau_D)),
 * with the explicit Runge-Kutta method of Dormand and Prince of order 5(4),
 * step size control, Shampine's dense output for the history, and steps
 * landing exactly on the breakpoints of dde_breakpoints (a C++ adaptation of
 * RETARD by Hairer and Wanner; see the notice at the top of this file).
 *
 * The integration starts with start(t0, z0, phi): the state z0 at t0, and the
 * constant initial history phi for t < t0. It then proceeds with
 * advance_to(t1), which steps exactly onto t1 and keeps the history, the step
 * size and the breakpoints for the next call. Between two calls the state may
 * jump (jump()) and the right hand side may change (change_rhs()), which is
 * how dosing events are integrated. Calling start() again forgets the history
 * and restarts the integration.
 *
 * As in RETARD, the step size never exceeds the smallest delay, so lagged
 * arguments always lie in the already computed past (or the initial
 * history). Every time at which the state or its first derivative is
 * discontinuous (t0, the jumps, the changes of the right hand side, and the
 * first propagation u + tau_j of a jump at u) starts a new segment of the
 * history, and a step only looks up the segment that contains its lagged
 * arguments. The lagged arguments of the stages at the end of a step that
 * ends on u + tau_j are therefore the left limits at u, and those of the
 * following step the right limits. The derivative of the solution jumps at
 * these times, so the stage derivative at the start of the following step is
 * recomputed instead of reusing the last stage of the step that ended there
 * (RETARD's IRTRN = 3 / 4 logic), if the lagged values jump or the system
 * uses derivatives of lagged arguments.
 *
 * <p>Sensitivities and jump conditions. If the system integrates forward
 * sensitivities, its state is laid out as z = (y, S_1, ..., S_P), where y has
 * num_states() components and S_p = dy / dtheta_p. A jump of y at a time u
 * that depends on theta moves the discontinuities it causes, and the
 * sensitivity equations, which only see the left and right limits of the
 * state, cannot account for that. The integrator applies the jump conditions
 *   S_p += (f^- - f^+) * d(time) / dtheta_p
 * where f^- and f^+ are the derivatives of y just before and after the
 * discontinuity:
 *  - at a jump of the state at u (jump()) and at a change of the right hand
 *    side at u (change_rhs()), with d(time) / dtheta = du / dtheta;
 *  - at u + tau_j for every jump of y at u, where the j-th lagged argument
 *    of the right hand side jumps, with
 *    d(time) / dtheta = du / dtheta + dtau_j / dtheta.
 * The second condition is needed even if u does not depend on theta: the
 * derivative of the dense output, which the sensitivity equations use for
 * dy(t - tau_j) / dtau_j, does not contain the delta function of a jump.
 * Several jumps at the same time are applied one after the other.
 *
 * @tparam System Type of the system. It must provide
 *   - <code>size_t size() const</code>, the number of components of z;
 *   - <code>size_t num_states() const</code>, the number N of components of
 *     y; size() must be a multiple of it, and P = size() / N - 1;
 *   - <code>size_t num_lag_derivatives() const</code>, the number m of
 *     leading components whose lagged derivatives the system needs;
 *   - <code>Eigen::MatrixXd delay_sensitivities() const</code>, the P x D
 *     matrix of dtau_j / dtheta_p;
 *   - <code>void operator()(double t, const Eigen::VectorXd& z,
 *       const Eigen::MatrixXd& z_lag, const Eigen::MatrixXd& z_lag_deriv,
 *       Eigen::VectorXd& dz_dt)</code>, where column j of the size() x D
 *     matrix z_lag is z(t - tau_j) and column j of the m x D matrix
 *     z_lag_deriv is the time derivative of the first m components at
 *     t - tau_j. The first N components of dz_dt may only depend on t, the
 *     first N components of z and the first N rows of z_lag.
 *   The integrator keeps a reference to the system. A caller that changes
 *   the right hand side of the system between two calls of advance_to() must
 *   do so through change_rhs().
 */
template <typename System>
class dde_retard_integrator {
 public:
  /**
   * @param function_name name of the calling function, for error messages
   * @param system right hand side of the system
   * @param tau positive delays
   * @param relative_tolerance relative tolerance of the local error
   * @param absolute_tolerance absolute tolerance of the local error
   * @param max_num_steps maximum number of steps (accepted or rejected)
   *   between two output times, and in one call of advance_to()
   */
  dde_retard_integrator(const char* function_name, System& system,
                        const std::vector<double>& tau,
                        double relative_tolerance, double absolute_tolerance,
                        long int max_num_steps)  // NOLINT(runtime/int)
      : function_name_(function_name),
        system_(system),
        tau_(tau),
        rtol_(relative_tolerance),
        atol_(absolute_tolerance),
        max_num_steps_(max_num_steps),
        n_(system.size()),
        num_states_(system.num_states()),
        num_params_(n_ / num_states_ - 1),
        m_(system.num_lag_derivatives()),
        delay_sensitivities_(system.delay_sensitivities()),
        tau_min_(std::numeric_limits<double>::infinity()),
        tau_max_(0),
        breakpoints_(tau_),
        history_(n_),
        segment_(tau_.size(), -1),
        t_start_(0),
        z_lag_(n_, tau_.size()),
        z_lag_deriv_(m_, tau_.size()),
        x_(0),
        y_(n_),
        k1_(n_),
        y1_(n_),
        ysti_(n_),
        k2_(n_),
        k3_(n_),
        k4_(n_),
        k5_(n_),
        k6_(n_),
        k7_(n_),
        c5_dense_(n_),
        err_vec_(n_),
        h_(0),
        facold_(1e-4),
        reject_(false),
        started_(false),
        k1_stale_(true),
        need_initial_step_(true) {
    for (double tau_j : tau_) {
      tau_min_ = std::min(tau_min_, tau_j);
      tau_max_ = std::max(tau_max_, tau_j);
    }
  }

  dde_retard_integrator(const dde_retard_integrator&) = delete;
  dde_retard_integrator& operator=(const dde_retard_integrator&) = delete;

  /**
   * Current time.
   */
  double time() const { return x_; }

  /**
   * Current state. At a time where the state is discontinuous this is the
   * right limit (after the jumps and jump conditions at that time).
   */
  const Eigen::VectorXd& state() const { return y_; }

  /**
   * Start (or restart) the integration at t0, forgetting the history.
   *
   * @param t0 initial time
   * @param z0 state at t0 (the right limit)
   * @param phi constant history for t < t0
   */
  void start(double t0, const Eigen::VectorXd& z0, const Eigen::VectorXd& phi) {
    check_size_match(function_name_, "state", z0.size(), "system", n_);
    check_size_match(function_name_, "history", phi.size(), "system", n_);
    t_start_ = t0;
    x_ = t0;
    y_ = z0;
    phi_ = phi;
    history_.clear();
    breakpoints_.restart(t0);
    std::fill(segment_.begin(), segment_.end(), -1);
    discontinuities_.clear();
    discontinuities_.emplace_back(t0, history_.end());
    discontinuity& d = discontinuities_.back();
    d.source = true;
    if (!(phi == z0)) {
      d.state_jump = true;
      if (!(phi.head(num_states_) == z0.head(num_states_))) {
        d.base_jump = true;
        d.states.push_back(phi);
        d.states.push_back(z0);
        d.time_sensitivities.emplace_back();
      }
    }
    started_ = true;
    k1_stale_ = true;
    need_initial_step_ = true;
    facold_ = 1e-4;
    reject_ = false;
  }

  /**
   * Integrate to t1 >= time(), stepping exactly onto t1.
   *
   * @param t1 end of this part of the integration
   * @throw std::domain_error if more than max_num_steps steps are needed to
   *   reach an output time or t1, or if the step size becomes too small
   *   (which is the case when the right hand side keeps returning non-finite
   *   values)
   */
  void advance_to(double t1) {
    advance_to(t1, std::vector<double>{},
               [](size_t, const Eigen::VectorXd&) {});
  }

  /**
   * Integrate to t1 >= time(), stepping exactly onto t1, and report the state
   * at the output times ts.
   *
   * @tparam Observer Type of the observer, called as
   *   <code>observer(i, z)</code> with the state z at ts[i] for each i in
   *   order.
   * @param t1 end of this part of the integration
   * @param ts output times, strictly increasing, greater than time() and not
   *   greater than t1
   * @param observer observer of the solution at the output times
   * @throw std::domain_error if more than max_num_steps steps are needed to
   *   reach an output time or t1, or if the step size becomes too small
   *   (which is the case when the right hand side keeps returning non-finite
   *   values)
   */
  template <typename Observer>
  void advance_to(double t1, const std::vector<double>& ts,
                  Observer&& observer) {
    // step size control parameters (RETARD defaults)
    static constexpr double uround = 10 * 2.3e-16;
    static constexpr double safe = 0.9;
    static constexpr double facc1 = 1.0 / 0.2;
    static constexpr double facc2 = 1.0 / 10.0;
    static constexpr double beta = 0.04;
    static constexpr double expo1 = 0.2 - beta * 0.75;

    if (!started_) {
      throw_domain_error(function_name_, "", x_,
                         "advance_to called before start at time (", ")");
    }
    if (t1 == x_) {
      return;
    }
    check_greater(function_name_, "end time", t1, x_);

    const double tol = 100 * std::numeric_limits<double>::epsilon()
                       * std::max(std::fabs(t_start_), std::fabs(t1));
    if (!(tau_min_ > tol)) {
      throw_domain_error(function_name_, "smallest delay", tau_min_, "is ",
                         ", but must be larger than 100 machine epsilons "
                         "times the largest absolute integration time");
    }
    // the maximal step size is the smallest delay (RETARD sets HMAX this way
    // during the first evaluation of the right hand side)
    const double hmax = std::min(t1 - x_, tau_min_);

    ensure_k1();
    if (need_initial_step_) {
      initial_step(hmax);
      need_initial_step_ = false;
    }

    bool last = false;
    double xend = breakpoints_.next(x_, t1, tol);
    size_t next_output = 0;
    long int num_steps = 0;  // NOLINT(runtime/int)
    double h = h_;

    while (true) {
      if (num_steps >= max_num_steps_) {
        throw_domain_error(function_name_, "",
                           next_output < ts.size() ? ts[next_output] : t1,
                           "Failed to integrate to next output time (",
                           ") in less than max_num_steps steps");
      }
      if (!(std::fabs(h) > std::fabs(x_) * uround)) {
        throw_domain_error(function_name_, "", x_,
                           "Failed to integrate: the step size became too "
                           "small at time (",
                           "); the right hand side is likely not finite");
      }
      if (x_ + 1.01 * h - xend > 0) {
        h = xend - x_;
        last = true;
      } else if (x_ + 1.8 * h - xend > 0) {
        h = (xend - x_) * 0.55;
      }
      const double xph = last ? xend : x_ + h;
      ++num_steps;

      const double err = dopri_step(h, xph);
      // a non-finite error estimate or state rejects the step and shrinks it
      // as much as allowed
      const bool finite = std::isfinite(err) && y1_.allFinite();

      // computation of hnew, with Lund stabilization, subject to
      // 0.2 <= hnew / h <= 10
      const double fac11 = std::pow(err, expo1);
      double hnew
          = h
            / std::max(facc2,
                       std::min(facc1, fac11 / std::pow(facold_, beta) / safe));
      if (finite && err <= 1.0) {
        // step is accepted
        facold_ = std::max(err, 1.0e-4);
        history_.push(x_, h, y_, y1_, k1_, k7_, c5_dense_);
        k1_.swap(k7_);
        y_.swap(y1_);
        x_ = xph;
        if (last) {
          land(x_ == t1, tol);
          hnew = 0.9 * hnew;
        }
        while (next_output < ts.size() && ts[next_output] <= x_) {
          if (ts[next_output] == x_) {
            observer(next_output, y_);
          } else {
            Eigen::VectorXd z_out(n_);
            const double s = ts[next_output];
            history_.value(history_.locate(s, 0, history_.end()), s, z_out);
            observer(next_output, z_out);
          }
          ++next_output;
          num_steps = 0;
        }
        history_.forget_before(x_ - tau_max_);
        if (std::fabs(hnew) > hmax) {
          hnew = hmax;
        }
        if (reject_) {
          hnew = std::min(std::fabs(hnew), std::fabs(h));
        }
        reject_ = false;
        h = hnew;
        if (last) {
          if (x_ == t1) {
            h_ = h;
            return;
          }
          last = false;
          xend = breakpoints_.next(x_, t1, tol);
        }
      } else {
        // step is rejected
        hnew = finite ? h / std::min(facc1, fac11 / safe) : h / facc1;
        reject_ = true;
        last = false;
        h = hnew;
      }
    }
  }

  /**
   * Add dz to the state at the current time, and apply the jump condition of
   * the sensitivities if the time of the jump depends on theta.
   *
   * @param dz jump of the state (including its sensitivity components)
   * @param du_dtheta derivatives of the time of the jump with respect to the
   *   P sensitivity parameters, or an empty vector if it does not depend on
   *   them
   */
  void jump(const Eigen::VectorXd& dz, const Eigen::VectorXd& du_dtheta) {
    check_size_match(function_name_, "state jump", dz.size(), "system", n_);
    check_time_sensitivities(du_dtheta);
    const bool moves = num_params_ > 0 && du_dtheta.size() > 0
                       && !(du_dtheta.array() == 0).all();
    Eigen::VectorXd f_left;
    if (moves) {
      ensure_k1();
      f_left = k1_.head(num_states_);
    }
    const Eigen::VectorXd z_left = y_;
    y_ += dz;
    discontinuity& d = discontinuity_here();
    const bool base_jump = !(dz.head(num_states_).array() == 0).all();
    if (!(dz.array() == 0).all() || moves) {
      d.state_jump = true;
    }
    if (base_jump) {
      if (!d.base_jump) {
        d.base_jump = true;
        d.states.push_back(z_left);
      }
      d.states.push_back(y_);
      d.time_sensitivities.push_back(du_dtheta);
    }
    if (moves) {
      Eigen::VectorXd f_right(n_);
      rhs(x_, y_, f_right);
      add_time_shift(f_left - f_right.head(num_states_), du_dtheta);
    }
    k1_stale_ = true;
    restart_step_control();
  }

  /**
   * Change the right hand side of the system at the current time, and apply
   * the jump condition of the sensitivities if that time depends on theta.
   *
   * @tparam Change Type of the change, a callable without arguments
   * @param du_dtheta derivatives of the current time with respect to the P
   *   sensitivity parameters, or an empty vector if it does not depend on
   *   them
   * @param change changes the system (for example its forcing) when called;
   *   the integrator evaluates the right hand side before and after it
   */
  template <typename Change>
  void change_rhs(const Eigen::VectorXd& du_dtheta, Change&& change) {
    check_time_sensitivities(du_dtheta);
    const bool moves = num_params_ > 0 && du_dtheta.size() > 0
                       && !(du_dtheta.array() == 0).all();
    Eigen::VectorXd f_left;
    if (moves) {
      ensure_k1();
      f_left = k1_.head(num_states_);
    }
    change();
    discontinuity& d = discontinuity_here();
    if (moves) {
      Eigen::VectorXd f_right(n_);
      rhs(x_, y_, f_right);
      add_time_shift(f_left - f_right.head(num_states_), du_dtheta);
      d.state_jump = true;
    }
    k1_stale_ = true;
    restart_step_control();
  }

 private:
  /**
   * A time at which the state or its first derivative is discontinuous. It
   * starts a new segment of the history.
   */
  struct discontinuity {
    discontinuity(double time, size_t first_step)
        : time(time),
          first_step(first_step),
          source(false),
          state_jump(false),
          base_jump(false) {}
    /// the time
    double time;
    /// global index of the first step of the segment it starts
    size_t first_step;
    /// whether its propagation is in the breakpoints
    bool source;
    /// whether some component of the state jumps
    bool state_jump;
    /// whether the first num_states() components of the state jump
    bool base_jump;
    /// for a base jump: the left limit, then the state after each jump
    std::vector<Eigen::VectorXd> states;
    /// for a base jump: the derivatives of the time of each jump
    std::vector<Eigen::VectorXd> time_sensitivities;
  };

  void check_time_sensitivities(const Eigen::VectorXd& du_dtheta) const {
    if (du_dtheta.size() > 0) {
      check_size_match(function_name_, "time sensitivities", du_dtheta.size(),
                       "sensitivity parameters", num_params_);
    }
  }

  /**
   * The discontinuity at the current time, which is created (and added to
   * the breakpoint sources) if there is none yet.
   */
  discontinuity& discontinuity_here() {
    if (discontinuities_.back().time != x_) {
      discontinuities_.emplace_back(x_, history_.end());
    }
    discontinuity& d = discontinuities_.back();
    if (!d.source) {
      breakpoints_.add_source(x_);
      d.source = true;
    }
    return d;
  }

  /**
   * Start a new step size sequence after a discontinuity at the current time.
   */
  void restart_step_control() {
    need_initial_step_ = true;
    facold_ = 1e-4;
    reject_ = false;
  }

  /**
   * S_p += df * dt_dtheta_p for p = 1, ..., P.
   */
  void add_time_shift(const Eigen::VectorXd& df,
                      const Eigen::VectorXd& dt_dtheta) {
    for (size_t p = 0; p < num_params_; ++p) {
      if (dt_dtheta.coeff(p) != 0) {
        y_.segment(num_states_ * (p + 1), num_states_)
            += dt_dtheta.coeff(p) * df;
      }
    }
  }

  /**
   * Lagged values and lagged derivatives for a stage at time t of the
   * current step.
   */
  void lags(double t, Eigen::MatrixXd& z_lag,
            Eigen::MatrixXd& z_lag_deriv) const {
    for (size_t j = 0; j < tau_.size(); ++j) {
      if (segment_[j] < 0) {
        z_lag.col(j) = phi_;
        z_lag_deriv.col(j).setZero();
      } else {
        const size_t seg = segment_[j];
        const size_t begin = discontinuities_[seg].first_step;
        const size_t end = seg + 1 < discontinuities_.size()
                               ? discontinuities_[seg + 1].first_step
                               : history_.end();
        const double s = t - tau_[j];
        const size_t k = history_.locate(s, begin, end);
        history_.value(k, s, z_lag.col(j));
        if (m_ > 0) {
          history_.derivative(k, s, m_, z_lag_deriv.col(j));
        }
      }
    }
  }

  void rhs(double t, const Eigen::VectorXd& z, Eigen::VectorXd& dz_dt) {
    lags(t, z_lag_, z_lag_deriv_);
    system_(t, z, z_lag_, z_lag_deriv_, dz_dt);
  }

  void ensure_k1() {
    if (k1_stale_) {
      rhs(x_, y_, k1_);
      k1_stale_ = false;
    }
  }

  /**
   * Initial step size (RETARD's HINIT, order 5).
   */
  void initial_step(double hmax) {
    const size_t n = n_;
    double dnf = 0;
    double dny = 0;
    for (size_t i = 0; i < n; ++i) {
      const double sk = atol_ + rtol_ * std::fabs(y_[i]);
      dnf += (k1_[i] / sk) * (k1_[i] / sk);
      dny += (y_[i] / sk) * (y_[i] / sk);
    }
    double h;
    if (dnf <= 1e-10 || dny <= 1e-10) {
      h = 1e-6;
    } else {
      h = std::sqrt(dny / dnf) * 0.01;
    }
    h = std::min(h, hmax);
    y1_ = y_ + h * k1_;
    rhs(x_ + h, y1_, k2_);
    double der2 = 0;
    for (size_t i = 0; i < n; ++i) {
      const double sk = atol_ + rtol_ * std::fabs(y_[i]);
      der2 += ((k2_[i] - k1_[i]) / sk) * ((k2_[i] - k1_[i]) / sk);
    }
    der2 = std::sqrt(der2) / h;
    const double der12 = std::max(std::fabs(der2), std::sqrt(dnf));
    double h1;
    if (der12 <= 1e-15) {
      h1 = std::max(1e-6, std::fabs(h) * 1e-3);
    } else {
      h1 = std::pow(0.01 / der12, 1.0 / 5.0);
    }
    h_ = std::min({100 * h, h1, hmax});
  }

  /**
   * One Dormand-Prince step from (x, y) with step size h, ending at xph.
   * Leaves the new state in y1, the last stage in k7 and the fifth dense
   * output coefficient in c5_dense, and returns the error estimate.
   */
  double dopri_step(double h, double xph) {
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

    const double x = x_;
    const Eigen::VectorXd& y = y_;
    const Eigen::VectorXd& k1 = k1_;
    // the stages; k1 is the derivative at (x, y)
    y1_ = y + h * a21 * k1;
    rhs(x + c2 * h, y1_, k2_);
    y1_ = y + h * (a31 * k1 + a32 * k2_);
    rhs(x + c3 * h, y1_, k3_);
    y1_ = y + h * (a41 * k1 + a42 * k2_ + a43 * k3_);
    rhs(x + c4 * h, y1_, k4_);
    y1_ = y + h * (a51 * k1 + a52 * k2_ + a53 * k3_ + a54 * k4_);
    rhs(x + c5 * h, y1_, k5_);
    ysti_ = y + h * (a61 * k1 + a62 * k2_ + a63 * k3_ + a64 * k4_ + a65 * k5_);
    rhs(xph, ysti_, k6_);
    y1_ = y + h * (a71 * k1 + a73 * k3_ + a74 * k4_ + a75 * k5_ + a76 * k6_);
    rhs(xph, y1_, k7_);
    c5_dense_
        = h * (d1 * k1 + d3 * k3_ + d4 * k4_ + d5 * k5_ + d6 * k6_ + d7 * k7_);
    err_vec_
        = h * (e1 * k1 + e3 * k3_ + e4 * k4_ + e5 * k5_ + e6 * k6_ + e7 * k7_);

    // error estimation
    double err = 0;
    for (size_t i = 0; i < n_; ++i) {
      const double sk
          = atol_ + rtol_ * std::max(std::fabs(y[i]), std::fabs(y1_[i]));
      const double r = err_vec_[i] / sk;
      err += r * r;
    }
    return std::sqrt(err / n_);
  }

  /**
   * Bookkeeping after a step landed on a breakpoint (or the end of the
   * integration) at the current time x: lagged arguments t - tau_j that
   * reach a discontinuity move on to its segment of the history, with the
   * jump conditions of the sensitivities, and the first stage derivative is
   * recomputed if it changes.
   *
   * @param at_end whether x is the end of this call of advance_to(); the
   *   first stage derivative is then only marked for recomputation
   * @param tol tolerance of the breakpoints
   */
  void land(bool at_end, double tol) {
    bool restart = false;
    bool new_discontinuity = false;
    bool new_state_jump = false;
    for (size_t j = 0; j < tau_.size(); ++j) {
      while (static_cast<size_t>(segment_[j] + 1) < discontinuities_.size()
             && discontinuities_[segment_[j] + 1].time + tau_[j] <= x_ + tol) {
        const size_t index = segment_[j] + 1;
        if (discontinuities_[index].base_jump) {
          restart = true;
          new_discontinuity = true;
          if (num_params_ > 0) {
            new_state_jump |= crossing_conditions(j, index);
          }
        } else if (discontinuities_[index].state_jump || m_ > 0) {
          restart = true;
        }
        segment_[j] = index;
      }
    }
    if (new_discontinuity) {
      // y' jumps where a lagged argument crosses a jump of y
      if (discontinuities_.back().time != x_) {
        discontinuities_.emplace_back(x_, history_.end());
      }
      discontinuities_.back().state_jump |= new_state_jump;
    }
    forget_discontinuities();
    if (restart) {
      if (at_end) {
        k1_stale_ = true;
      } else {
        rhs(x_, y_, k1_);
      }
    }
  }

  /**
   * Jump conditions where the lagged argument t - tau_j crosses the jumps of
   * y at the discontinuity with the given index, at the current time.
   *
   * @return whether the state changed
   */
  bool crossing_conditions(size_t j, size_t index) {
    const discontinuity& d = discontinuities_[index];
    bool changed = false;
    Eigen::MatrixXd z_lag(n_, tau_.size());
    Eigen::MatrixXd z_lag_deriv(m_, tau_.size());
    Eigen::VectorXd f_left(n_);
    Eigen::VectorXd f_right(n_);
    for (size_t k = 1; k < d.states.size(); ++k) {
      Eigen::VectorXd dt_dtheta = delay_sensitivities_.col(j);
      if (d.time_sensitivities[k - 1].size() > 0) {
        dt_dtheta += d.time_sensitivities[k - 1];
      }
      if ((dt_dtheta.array() == 0).all()) {
        continue;
      }
      lags(x_, z_lag, z_lag_deriv);
      z_lag.col(j) = d.states[k - 1];
      system_(x_, y_, z_lag, z_lag_deriv, f_left);
      z_lag.col(j) = d.states[k];
      system_(x_, y_, z_lag, z_lag_deriv, f_right);
      add_time_shift(f_left.head(num_states_) - f_right.head(num_states_),
                     dt_dtheta);
      changed = true;
    }
    return changed;
  }

  /**
   * Drop the discontinuities that no lagged argument can reach any more.
   */
  void forget_discontinuities() {
    if (tau_.empty()) {
      return;
    }
    const int reached = *std::min_element(segment_.begin(), segment_.end());
    if (reached > 0) {
      discontinuities_.erase(discontinuities_.begin(),
                             discontinuities_.begin() + reached);
      for (int& seg : segment_) {
        seg -= reached;
      }
    }
  }

  const char* function_name_;
  System& system_;
  const std::vector<double> tau_;
  const double rtol_;
  const double atol_;
  const long int max_num_steps_;  // NOLINT(runtime/int)
  const size_t n_;
  const size_t num_states_;
  const size_t num_params_;
  const size_t m_;
  const Eigen::MatrixXd delay_sensitivities_;
  double tau_min_;
  double tau_max_;
  dde_breakpoints breakpoints_;
  dde_dense_history history_;
  std::vector<discontinuity> discontinuities_;
  /// index of the segment of the history that lagged argument j is in, or -1
  /// for the initial history
  std::vector<int> segment_;
  double t_start_;
  Eigen::VectorXd phi_;
  Eigen::MatrixXd z_lag_;
  Eigen::MatrixXd z_lag_deriv_;
  double x_;
  Eigen::VectorXd y_;
  Eigen::VectorXd k1_;
  Eigen::VectorXd y1_;
  Eigen::VectorXd ysti_;
  Eigen::VectorXd k2_;
  Eigen::VectorXd k3_;
  Eigen::VectorXd k4_;
  Eigen::VectorXd k5_;
  Eigen::VectorXd k6_;
  Eigen::VectorXd k7_;
  Eigen::VectorXd c5_dense_;
  Eigen::VectorXd err_vec_;
  double h_;
  double facold_;
  bool reject_;
  bool started_;
  bool k1_stale_;
  bool need_initial_step_;
};

}  // namespace internal
}  // namespace math
}  // namespace stan
#endif
