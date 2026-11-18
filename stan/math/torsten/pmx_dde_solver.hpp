#ifndef STAN_MATH_TORSTEN_PMX_DDE_SOLVER_HPP
#define STAN_MATH_TORSTEN_PMX_DDE_SOLVER_HPP

#include <stan/math/prim/functor/dde_retard.hpp>
#include <stan/math/rev/functor/solve_powell.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/torsten/dsolve/pmx_dde_system.hpp>
#include <stan/math/torsten/pmx_dde_schedule.hpp>
#include <Eigen/Dense>
#include <deque>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace torsten {

  /**
   * Control parameters of <code>pmx_solve_dde_rk45</code>: tolerances and
   * maximum number of steps of the DDE integrator, and the same for the
   * steady-state solves (in units of dosing periods for the burn-in).
   */
  struct PMXDdeControls {
    double rtol = 1.e-6;
    double atol = 1.e-6;
    long int max_num_steps = 1000000;  // NOLINT(runtime/int)
    double as_rtol = 1.e-6;
    double as_atol = 1.e-6;
    long int as_max_num_steps = 100;  // NOLINT(runtime/int)
    std::ostream* msgs = nullptr;
  };

  /**
   * Solves a PMX DDE over the events of one subject.
   *
   * One resumable integrator (<code>dde_retard_integrator</code>) lives
   * across all events, so the history of the solution, and of its
   * sensitivities, survives the doses. The events map onto the integrator's
   * operations:
   *  - between two events the integrator advances to the next event time,
   *    after a change of the right hand side if the new interval has another
   *    parameter row, other rates or other data;
   *  - a bolus dose is a jump of the state;
   *  - turning a compartment off (EVID 2, negative CMT) and overwriting it
   *    are jumps to the new value;
   *  - a reset (EVID 3, and EVID 4 before its dose) restarts the integration
   *    from the history <code>y_hist</code>, forgetting the earlier solution;
   *  - a steady-state dose (SS = 1) restarts from the periodic steady state
   *    of the dosing regimen.
   *
   * The forward sensitivities are taken with respect to the global
   * parameters of the schedule. The times of jumps, of changes of the right
   * hand side and of restarts may depend on them; the integrator applies the
   * jump conditions of a moving discontinuity, and a restart at time u
   * starts the sensitivities from dy_hist/dp - f(u) du/dp.
   *
   * @tparam F type of the user's right hand side
   */
  template <typename F>
  class PMXDdeSolver {
    using system_t = dsolve::PMXDdeSystem<F>;
    using integrator_t = stan::math::internal::dde_retard_integrator<system_t>;

  public:
    static constexpr const char* function_name = "pmx_solve_dde_rk45";

    /**
     * @param f the user's right hand side
     * @param ncmt number of compartments
     * @param tau delays
     * @param delay_sensitivities P x D derivatives of the delays
     * @param y_hist history: the state before the first event and after a
     *   reset
     * @param y_hist_sensitivities nCmt x P derivatives of the history
     * @param controls tolerances and step limits
     */
    PMXDdeSolver(const F& f, int ncmt, const std::vector<double>& tau,
                 const Eigen::MatrixXd& delay_sensitivities,
                 const Eigen::VectorXd& y_hist,
                 const Eigen::MatrixXd& y_hist_sensitivities,
                 const PMXDdeControls& controls)
      : f_(f),
        n_(ncmt),
        p_(y_hist_sensitivities.cols()),
        tau_(tau),
        delay_sensitivities_(delay_sensitivities),
        controls_(controls),
        y_hist_(y_hist),
        z_hist_(n_ * (1 + p_)),
        system_(f, ncmt, p_, delay_sensitivities, true, controls.msgs),
        integrator_(function_name, system_, tau, controls.rtol, controls.atol,
                    controls.max_num_steps) {
      z_hist_.head(n_) = y_hist;
      for (size_t p = 0; p < p_; ++p) {
        z_hist_.segment(n_ * (1 + p), n_) = y_hist_sensitivities.col(p);
      }
    }

    /**
     * Solve the events.
     *
     * @param events the events of the subject, in order
     * @param[out] values nCmt x (number of kept events) state after each
     *   kept event
     * @param[out] sensitivities for each kept event, the nCmt x P
     *   derivatives of its state with respect to the global parameters
     */
    void solve(const std::vector<PMXDdeEvent>& events, Eigen::MatrixXd& values,
               std::vector<Eigen::MatrixXd>& sensitivities) {
      int nkeep = 0;
      for (const PMXDdeEvent& e : events) {
        nkeep += e.keep;
      }
      values.resize(n_, nkeep);
      sensitivities.clear();
      sensitivities.reserve(nkeep);

      system_.set_rhs(events[0].rhs);
      restart(events[0].time, events[0].time_sensitivities);
      int ikeep = 0;
      for (size_t i = 0; i < events.size(); ++i) {
        const PMXDdeEvent& e = events[i];
        const bool restarts = e.id >= 1 && e.id <= 4;
        if (i > 0) {
          if (restarts) {
            system_.set_rhs(e.rhs);
          } else {
            if (system_.rhs() != e.rhs) {
              integrator_.change_rhs(events[i - 1].time_sensitivities,
                                     [&] { system_.set_rhs(e.rhs); });
            }
            integrator_.advance_to(e.time);
          }
        }

        switch (e.id) {
        case 1:  // EVID 3: reset
          restart(e.time, e.time_sensitivities);
          break;
        case 2:  // EVID 4: reset and dose
          restart(e.time, e.time_sensitivities);
          dose(e);
          break;
        case 3:  // SS = 2
          throw std::domain_error(
              std::string(function_name)
              + ": steady-state dosing with SS = 2 (adding a steady state to "
                "the current state) is not supported: superposition does not "
                "hold for nonlinear systems, and is undefined for the history "
                "of a DDE");
        case 4:  // SS = 1
          steady_state(e);
          dose(e);
          break;
        case 5:  // EVID 2, negative CMT: turn the compartment off
          overwrite(-e.cmt - 1, 0.0, Eigen::VectorXd(), e.time_sensitivities);
          break;
        case 6:  // overwrite a compartment
          overwrite(e.cmt - 1, e.amount, e.amount_sensitivities,
                    e.time_sensitivities);
          break;
        default:  // transient dose or no action
          dose(e);
        }

        if (e.keep) {
          const Eigen::VectorXd& z = integrator_.state();
          values.col(ikeep) = z.head(n_);
          Eigen::MatrixXd s = Eigen::Map<const Eigen::MatrixXd>(z.data() + n_, n_, p_);
          if (e.time_sensitivities.size() > 0) {
            // the state after the event moves with the event time: its total
            // derivative is S + f dt/dp, with f the right limit after the
            // event (equal to S^- + f^- dt/dp before it, by the jump
            // conditions of the event's actions)
            s += current_derivative() * e.time_sensitivities.transpose();
          }
          sensitivities.push_back(s);
          ++ikeep;
        }
      }
    }

  private:
    /**
     * Restart the integration at time t from the history, forgetting the
     * earlier solution. The solution after t depends on t: its derivative
     * with respect to t just after the restart is -f(t, y_hist, y_hist), so
     * the sensitivities start from dy_hist/dp - f dt/dp. The history before
     * t is constant and keeps the sensitivities dy_hist/dp.
     *
     * The right hand side used for f is the current one. Should it change at
     * t (the next interval has another one), the integrator's jump condition
     * of that change replaces it by the new one.
     */
    void restart(double t, const Eigen::VectorXd& dt_dtheta) {
      Eigen::VectorXd z0 = z_hist_;
      if (dt_dtheta.size() > 0) {
        const Eigen::MatrixXd y_lag = y_hist_.replicate(1, tau_.size());
        const Eigen::VectorXd f = system_.base(t, y_hist_, y_lag);
        for (size_t p = 0; p < p_; ++p) {
          z0.segment(n_ * (1 + p), n_) -= dt_dtheta.coeff(p) * f;
        }
      }
      integrator_.start(t, z0, z_hist_);
    }

    /**
     * Add the bolus amount of an event to its compartment.
     */
    void dose(const PMXDdeEvent& e) {
      if (e.amount == 0 && e.amount_sensitivities.size() == 0) {
        return;
      }
      const int c = e.cmt - 1;
      Eigen::VectorXd dz = Eigen::VectorXd::Zero(integrator_.state().size());
      dz.coeffRef(c) = e.amount;
      if (e.amount_sensitivities.size() > 0) {
        for (size_t p = 0; p < p_; ++p) {
          dz.coeffRef(n_ * (1 + p) + c) = e.amount_sensitivities.coeff(p);
        }
      }
      integrator_.jump(dz, e.time_sensitivities);
    }

    /**
     * Set compartment c to the given amount.
     *
     * The jump y_c -> amount depends on the state, so if its time u moves
     * with the parameters the sensitivity of compartment c just after it is
     * d(amount)/dp - f_c(u+) du/dp. The integrator adds
     * (f(u-) - f(u+)) du/dp to every component, so the jump of the
     * sensitivity passed to it also removes S_c + f_c(u-) du/dp.
     */
    void overwrite(int c, double amount, const Eigen::VectorXd& amount_sensitivities,
                   const Eigen::VectorXd& dt_dtheta) {
      const Eigen::VectorXd& z = integrator_.state();
      Eigen::VectorXd dz = Eigen::VectorXd::Zero(z.size());
      dz.coeffRef(c) = amount - z.coeff(c);
      for (size_t p = 0; p < p_; ++p) {
        dz.coeffRef(n_ * (1 + p) + c) = -z.coeff(n_ * (1 + p) + c);
      }
      if (amount_sensitivities.size() > 0) {
        for (size_t p = 0; p < p_; ++p) {
          dz.coeffRef(n_ * (1 + p) + c) += amount_sensitivities.coeff(p);
        }
      }
      if (dt_dtheta.size() > 0) {
        const double f_c = current_derivative()(c);
        for (size_t p = 0; p < p_; ++p) {
          dz.coeffRef(n_ * (1 + p) + c) -= f_c * dt_dtheta.coeff(p);
        }
      }
      integrator_.jump(dz, dt_dtheta);
    }

    /**
     * The derivative of the base state at the current time and state, with
     * the integrator's lagged values: the right limit where it jumps, and
     * before a jump or a change of the right hand side at the current time
     * the left limit f^- that the jump conditions use.
     */
    Eigen::VectorXd current_derivative() {
      return integrator_.derivative().head(n_);
    }

    /**
     * Restart the integration at a steady-state dose (SS = 1) from the steady
     * state of its regimen, the state just before the dose after an infinite
     * past of the same dose. The bolus part of the dose itself is added by
     * the caller.
     *
     * The right hand side of the regimen is that of the dosing event, with
     * only the regimen's own infusions; infusions of the regimen still
     * running at the dose are dropped afterwards, as in Torsten's ODE
     * solvers. An infusion has rate RATE x F and duration AMT / RATE, as for
     * transient doses.
     */
    void steady_state(const PMXDdeEvent& e) {
      ss_rhs_ = e.rhs;
      ss_rhs_.rate.setZero();
      ss_rhs_.rate_sensitivities.resize(0, 0);
      if (e.ii > 0) {
        periodic_steady_state(e);
      } else if (e.ss_rate > 0) {
        constant_infusion_steady_state(e);
      } else {
        throw std::domain_error(std::string(function_name)
                                + ": steady-state bolus dosing (SS = 1, RATE = 0) "
                                  "needs a dosing interval II > 0");
      }
    }

    /**
     * Periodic steady state (II > 0), by burn-in: the regimen is integrated
     * from the history y_hist for as many dosing periods as it takes the
     * state at successive doses to agree within as_rtol / as_atol, at most
     * as_max_num_steps periods.
     *
     * The number of periods K is found on the base system, without
     * sensitivities. The burn-in that is kept integrates the sensitivities
     * over the K periods that end at the steady-state dose, so that its
     * history is that of the regimen in the time frame of the schedule, and
     * the derivatives with respect to the parameters, including the dose
     * time and the interval, converge together with the state when the
     * period map contracts.
     */
    void periodic_steady_state(const PMXDdeEvent& e) {
      const int max_periods = controls_.as_max_num_steps;
      system_t base(f_, n_, p_, delay_sensitivities_, false, controls_.msgs);
      dsolve::PMXDdeRhs base_rhs = ss_rhs_;
      base.set_rhs(base_rhs);
      integrator_t base_integrator(function_name, base, tau_, controls_.rtol,
                                   controls_.atol, controls_.max_num_steps);
      const double t_end = e.time + max_periods * e.ii;
      base_integrator.start(t_end - max_periods * e.ii, y_hist_, y_hist_);
      Eigen::VectorXd previous = y_hist_;
      const int periods = regimen(
          base_integrator, base_rhs, e, t_end, max_periods, false, [&](int) {
            const Eigen::VectorXd& x = base_integrator.state();
            const bool converged
                = ((x - previous).array().abs()
                   <= controls_.as_atol + controls_.as_rtol * x.array().abs())
                      .all();
            previous = x;
            return converged;
          });
      if (periods < 0) {
        throw std::domain_error(
            std::string(function_name)
            + ": the steady state was not reached: the states at successive "
              "doses still differ by more than as_rtol / as_atol after "
              "as_max_num_steps = " + std::to_string(max_periods)
            + " dosing periods (the regimen may not have an attracting "
              "periodic solution)");
      }

      system_.set_rhs(ss_rhs_);
      restart(e.time - periods * e.ii,
              add_scaled(e.time_sensitivities, -periods, e.ii_sensitivities));
      regimen(integrator_, ss_rhs_, e, e.time, periods, true,
              [](int) { return false; });
    }

    /**
     * Drive an integrator through n periods of the regimen of a steady-state
     * dose, with doses at t_end - (n - k) II, k = 0, ..., n - 1, from the
     * first dose time (where the integrator must be) to t_end.
     *
     * @param integrator the integrator
     * @param rhs the right hand side of the integrator's system, whose rates
     *   are set here (zero at the start)
     * @param e the steady-state dosing event
     * @param t_end end of the last period
     * @param n number of periods
     * @param sensitivities whether the system has sensitivities
     * @param stop called with k at the k-th dose time (before the dose), k =
     *   1, ..., n; the burn-in stops when it returns true
     * @return the number of periods after which stop returned true, or -1
     */
    template <typename Integrator, typename Stop>
    int regimen(Integrator& integrator, dsolve::PMXDdeRhs& rhs,
                const PMXDdeEvent& e, double t_end, int n, bool sensitivities,
                Stop&& stop) {
      const int c = e.cmt - 1;
      int active = 0;
      // ends of the running infusions, with their time sensitivities
      std::deque<std::pair<double, Eigen::VectorXd> > ends;
      auto set_rate = [&](int m) {
        rhs.rate.coeffRef(c) = m * e.ss_rate;
        if (sensitivities && e.ss_rate_sensitivities.size() > 0) {
          rhs.rate_sensitivities = Eigen::MatrixXd::Zero(n_, p_);
          rhs.rate_sensitivities.row(c) = m * e.ss_rate_sensitivities.transpose();
        }
      };
      auto advance = [&](double t) {
        while (!ends.empty() && ends.front().first <= t) {
          integrator.advance_to(ends.front().first);
          integrator.change_rhs(ends.front().second, [&] { set_rate(--active); });
          ends.pop_front();
        }
        integrator.advance_to(t);
      };
      for (int k = 0; k < n; ++k) {
        const double t = t_end - (n - k) * e.ii;
        advance(t);
        if (k > 0 && stop(k)) {
          return k;
        }
        const Eigen::VectorXd dt
            = sensitivities ? add_scaled(e.time_sensitivities, k - n, e.ii_sensitivities)
                            : Eigen::VectorXd();
        if (e.ss_rate == 0) {
          Eigen::VectorXd dz = Eigen::VectorXd::Zero(integrator.state().size());
          dz.coeffRef(c) = e.ss_amount;
          if (sensitivities && e.ss_amount_sensitivities.size() > 0) {
            for (size_t p = 0; p < p_; ++p) {
              dz.coeffRef(n_ * (1 + p) + c) = e.ss_amount_sensitivities.coeff(p);
            }
          }
          integrator.jump(dz, dt);
        } else if (e.ss_duration > 0) {
          integrator.change_rhs(dt, [&] { set_rate(++active); });
          ends.emplace_back(t + e.ss_duration,
                            sensitivities ? add_scaled(dt, 1, e.ss_duration_sensitivities)
                                          : Eigen::VectorXd());
        }
      }
      advance(t_end);
      return stop(n) ? n : -1;
    }

    /**
     * Steady state of a constant infusion (II = 0): the equilibrium x of
     * f(t, x, [x, ..., x]) + rate = 0, found from y_hist by Powell's dogleg
     * method, whose trust region handles a guess far from the equilibrium
     * (the history is typically the drug-free state), with derivatives from
     * the implicit function theorem. The history of an equilibrium is
     * constant, so the integration restarts from it.
     */
    void constant_infusion_steady_state(const PMXDdeEvent& e) {
      if (e.ss_amount > 0) {
        throw std::domain_error(std::string(function_name)
                                + ": a constant-infusion steady state (SS = 1, "
                                  "II = 0, RATE > 0) needs AMT = 0");
      }
      const int c = e.cmt - 1;
      ss_rhs_.rate.coeffRef(c) = e.ss_rate;
      if (e.ss_rate_sensitivities.size() > 0) {
        ss_rhs_.rate_sensitivities = Eigen::MatrixXd::Zero(n_, p_);
        ss_rhs_.rate_sensitivities.row(c) = e.ss_rate_sensitivities.transpose();
      }
      system_.set_rhs(ss_rhs_);
      const double t = e.time;
      Eigen::VectorXd x;
      try {
        x = stan::math::solve_powell_tol(
            [&](const auto& u, std::ostream* msgs) {
              return system_.equilibrium_residual(t, u);
            },
            y_hist_, controls_.as_rtol, controls_.as_atol,
            controls_.as_max_num_steps, controls_.msgs);
      } catch (const std::exception& err) {
        throw std::domain_error(std::string(function_name)
                                + ": failed to find the steady state of a "
                                  "constant infusion: "
                                + err.what());
      }
      Eigen::VectorXd z = Eigen::VectorXd::Zero(n_ * (1 + p_));
      z.head(n_) = x;
      if (p_ > 0) {
        Eigen::MatrixXd jac_x, jac_p;
        system_.equilibrium_jacobian(t, x, jac_x, jac_p);
        const Eigen::FullPivLU<Eigen::MatrixXd> lu(jac_x);
        if (!lu.isInvertible()) {
          throw std::domain_error(std::string(function_name)
                                  + ": the steady state of a constant infusion "
                                    "is not isolated (singular Jacobian)");
        }
        const Eigen::MatrixXd s = -lu.solve(jac_p);
        for (size_t p = 0; p < p_; ++p) {
          z.segment(n_ * (1 + p), n_) = s.col(p);
        }
      }
      integrator_.start(t, z, z);
    }

    /**
     * a + s b for derivative vectors, where an empty vector is zero.
     */
    static Eigen::VectorXd add_scaled(const Eigen::VectorXd& a, double s,
                                      const Eigen::VectorXd& b) {
      if (b.size() == 0 || s == 0) {
        return a;
      }
      if (a.size() == 0) {
        return s * b;
      }
      return a + s * b;
    }

    const F& f_;
    const size_t n_;
    const size_t p_;
    const std::vector<double> tau_;
    const Eigen::MatrixXd delay_sensitivities_;
    const PMXDdeControls controls_;
    const Eigen::VectorXd y_hist_;
    Eigen::VectorXd z_hist_;
    system_t system_;
    integrator_t integrator_;
    /// right hand side of the regimen of the current steady-state dose
    dsolve::PMXDdeRhs ss_rhs_;
  };

}

#endif
