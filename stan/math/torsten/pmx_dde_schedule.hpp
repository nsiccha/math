#ifndef STAN_MATH_TORSTEN_PMX_DDE_SCHEDULE_HPP
#define STAN_MATH_TORSTEN_PMX_DDE_SCHEDULE_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/rev/meta.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/torsten/ev_manager.hpp>
#include <stan/math/torsten/ev_record.hpp>
#include <stan/math/torsten/dsolve/pmx_dde_system.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace torsten {

  /**
   * The global parameters of a PMX DDE solve: the distinct autodiff
   * variables among all its inputs, each counted once however often it
   * appears (a 2-d theta built from repeated rows costs no more than a 1-d
   * one). The forward sensitivities of the DDE are taken with respect to
   * them.
   */
  class PMXDdeParameters {
  public:
    void add(double) {}

    void add(const stan::math::var& x) {
      if (index_.emplace(x.vi_, varis_.size()).second) {
        varis_.push_back(x.vi_);
      }
    }

    template <typename T>
    void add(const std::vector<T>& v) {
      for (const auto& x : v) {
        add(x);
      }
    }

    template <typename T, int R, int C>
    void add(const Eigen::Matrix<T, R, C>& m) {
      for (Eigen::Index i = 0; i < m.size(); ++i) {
        add(m.coeff(i));
      }
    }

    /**
     * Index of a variable among the global parameters.
     */
    int index(const stan::math::var& x) const { return index_.at(x.vi_); }

    int index(double) const { return -1; }

    size_t size() const { return varis_.size(); }

    /**
     * The parameters as operands of the result.
     */
    std::vector<stan::math::var> operands() const {
      std::vector<stan::math::var> res;
      res.reserve(varis_.size());
      for (stan::math::vari* vi : varis_) {
        res.emplace_back(vi);
      }
      return res;
    }

    double value(size_t p) const { return varis_[p]->val_; }

  private:
    std::vector<stan::math::vari*> varis_;
    std::unordered_map<stan::math::vari*, size_t> index_;
  };

  /**
   * Fresh copies of the global parameters on a nested autodiff stack, so
   * that the derivatives of quantities Torsten's event handling computes from
   * the inputs (event times, amounts, rates) can be taken by nested reverse
   * sweeps. Must be constructed and used within one nested scope.
   */
  class PMXDdeLocalParameters {
  public:
    explicit PMXDdeLocalParameters(const PMXDdeParameters& params)
      : params_(params) {
      local_.reserve(params.size());
      for (size_t p = 0; p < params.size(); ++p) {
        local_.emplace_back(params.value(p));
        index_.emplace(local_.back().vi_, p);
      }
    }

    /**
     * Copy of an input with each autodiff variable replaced by its local
     * copy.
     */
    stan::math::var copy(const stan::math::var& x) const {
      return local_[params_.index(x)];
    }

    double copy(double x) const { return x; }

    template <typename T>
    std::vector<T> copy(const std::vector<T>& v) const {
      std::vector<T> res;
      res.reserve(v.size());
      for (const auto& x : v) {
        res.push_back(copy(x));
      }
      return res;
    }

    /**
     * Index of the global parameter a local variable is a copy of, or -1 if
     * it is none (data, or a quantity computed from the parameters).
     */
    int index(const stan::math::var& x) const {
      auto it = index_.find(x.vi_);
      return it == index_.end() ? -1 : static_cast<int>(it->second);
    }

    int index(double) const { return -1; }

    /**
     * Derivatives of a quantity computed from the local copies with respect
     * to the global parameters, or an empty vector if they are all zero.
     */
    Eigen::VectorXd gradient(const stan::math::var& x) {
      Eigen::VectorXd g = Eigen::VectorXd::Zero(local_.size());
      const int p = index(x);
      if (p >= 0) {
        g.coeffRef(p) = 1;
        return g;
      }
      stan::math::set_zero_all_adjoints_nested();
      stan::math::grad(x.vi_);
      for (size_t q = 0; q < local_.size(); ++q) {
        g.coeffRef(q) = local_[q].adj();
      }
      stan::math::set_zero_all_adjoints_nested();
      if (g.isZero()) {
        return Eigen::VectorXd();
      }
      return g;
    }

    Eigen::VectorXd gradient(double) { return Eigen::VectorXd(); }

  private:
    const PMXDdeParameters& params_;
    std::vector<stan::math::var> local_;
    std::unordered_map<stan::math::vari*, size_t> index_;
  };

  /**
   * One event of a PMX DDE schedule, after Torsten's event handling (addl
   * doses, infusion ends, lag times, bioavailability) and with all
   * quantities reduced to values and derivatives with respect to the global
   * parameters. Empty derivative vectors mean the quantity is data.
   */
  struct PMXDdeEvent {
    /// Torsten's event id, see <code>Event</code>
    int id;
    /// compartment of the event (negative: turn the compartment off)
    int cmt;
    /// whether the event is in the user's schedule (its state is returned)
    bool keep;
    /// time of the event
    double time;
    Eigen::VectorXd time_sensitivities;
    /// bolus amount added at the event (bioavailability included)
    double amount;
    Eigen::VectorXd amount_sensitivities;
    /// right hand side on the interval that ends at this event
    dsolve::PMXDdeRhs rhs;
    /// steady-state dosing: SS flag, interval, amount (bioavailability
    /// included), rate (bioavailability included) and infusion duration
    int ss;
    double ii;
    Eigen::VectorXd ii_sensitivities;
    double ss_amount;
    Eigen::VectorXd ss_amount_sensitivities;
    double ss_rate;
    Eigen::VectorXd ss_rate_sensitivities;
    double ss_duration;
    Eigen::VectorXd ss_duration_sensitivities;
  };

  /**
   * The events of a PMX DDE schedule, generated by Torsten's
   * <code>EventsManager</code> exactly as for <code>pmx_solve_rk45</code>.
   *
   * The events manager runs on local copies of the parameters on a nested
   * autodiff stack, and the derivatives of every event quantity with respect
   * to the global parameters are taken there.
   */
  template <typename T0, typename T1, typename T2, typename T3, typename T4,
            typename T5, typename T6>
  std::vector<PMXDdeEvent>
  pmx_dde_events(int ncmt,
                 const std::vector<T0>& time,
                 const std::vector<T1>& amt,
                 const std::vector<T2>& rate,
                 const std::vector<T3>& ii,
                 const std::vector<int>& evid,
                 const std::vector<int>& cmt,
                 const std::vector<int>& addl,
                 const std::vector<int>& ss,
                 const std::vector<std::vector<T4> >& theta,
                 const std::vector<std::vector<T5> >& biovar,
                 const std::vector<std::vector<T6> >& tlag,
                 const std::vector<std::vector<double> >& x_r,
                 const std::vector<std::vector<int> >& x_i,
                 const PMXDdeParameters& params) {
    using stan::math::value_of;
    using ER = NONMENEventsRecord<T0, T1, T2, T3>;
    using EM = EventsManager<ER, NonEventParameters<T0, T4, std::vector,
                                                    std::tuple<T5, T6>, double, int> >;

    stan::math::nested_rev_autodiff nested;
    PMXDdeLocalParameters local(params);

    const std::vector<T0> time_l = local.copy(time);
    const std::vector<T1> amt_l = local.copy(amt);
    const std::vector<T2> rate_l = local.copy(rate);
    const std::vector<T3> ii_l = local.copy(ii);
    const std::vector<std::vector<T4> > theta_l = local.copy(theta);
    const std::vector<std::vector<T5> > biovar_l = local.copy(biovar);
    const std::vector<std::vector<T6> > tlag_l = local.copy(tlag);

    const ER rec(ncmt, time_l, amt_l, rate_l, ii_l, evid, cmt, addl, ss);
    const EM em(0, rec, theta_l, biovar_l, tlag_l, x_r, x_i);
    const auto& events = em.events();

    const double eps = 1.0E-12;
    std::vector<PMXDdeEvent> res(events.size());
    for (size_t i = 0; i < events.size(); ++i) {
      PMXDdeEvent& e = res[i];
      const auto ev = em.event(i);
      e.id = ev.id;
      e.cmt = ev.cmt;
      e.keep = events.keep(i);
      e.time = value_of(ev.t1);
      e.time_sensitivities = local.gradient(ev.t1);

      // the bolus amount, as in Event::operator()
      e.amount = 0;
      if (ev.force0 < eps && ev.cmt != 0) {
        const auto& jp = ev.jump(std::abs(ev.cmt) - 1);
        e.amount = value_of(jp);
        e.amount_sensitivities = local.gradient(jp);
      }

      const auto& theta_i = em.theta(i);
      e.rhs.theta.resize(theta_i.size());
      e.rhs.theta_index.resize(theta_i.size());
      for (size_t k = 0; k < theta_i.size(); ++k) {
        e.rhs.theta[k] = value_of(theta_i[k]);
        e.rhs.theta_index[k] = local.index(theta_i[k]);
      }
      e.rhs.rate.resize(ncmt);
      for (int j = 0; j < ncmt; ++j) {
        e.rhs.rate.coeffRef(j) = value_of(ev.force[j]);
        const Eigen::VectorXd g = local.gradient(ev.force[j]);
        if (g.size() > 0) {
          if (e.rhs.rate_sensitivities.size() == 0) {
            e.rhs.rate_sensitivities = Eigen::MatrixXd::Zero(ncmt, params.size());
          }
          e.rhs.rate_sensitivities.row(j) = g.transpose();
        }
      }
      e.rhs.x_r = &em.template get_model_array_1d_param<0>(i);
      e.rhs.x_i = &em.template get_model_array_1d_param<1>(i);

      e.ss = events.ss(i);
      e.ii = value_of(ev.ii);
      e.ii_sensitivities = local.gradient(ev.ii);
      e.ss_amount = 0;
      e.ss_rate = 0;
      e.ss_duration = 0;
      if (events.is_ss_dosing(i)) {
        // Torsten's transient semantics of an infusion: the rate is the
        // dosing rate times the bioavailability, its duration amt / rate
        const auto& f_cmt = em.params.bioavailability(i, ev.cmt - 1);
        const auto ss_amount = em.fractioned_amt(i);
        e.ss_amount = value_of(ss_amount);
        e.ss_amount_sensitivities = local.gradient(ss_amount);
        const auto ss_rate = events.rate(i) * f_cmt;
        e.ss_rate = value_of(ss_rate);
        e.ss_rate_sensitivities = local.gradient(ss_rate);
        if (events.rate(i) > 0 && events.amt(i) > 0) {
          const auto duration = events.amt(i) / events.rate(i);
          e.ss_duration = value_of(duration);
          e.ss_duration_sensitivities = local.gradient(duration);
        }
      }
    }
    return res;
  }

}

#endif
