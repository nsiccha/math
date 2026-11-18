#ifndef STAN_MATH_TORSTEN_DSOLVE_PMX_DDE_SYSTEM_HPP
#define STAN_MATH_TORSTEN_DSOLVE_PMX_DDE_SYSTEM_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/rev/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <Eigen/Dense>
#include <ostream>
#include <vector>

namespace torsten {
namespace dsolve {

  /**
   * The right hand side of a PMX DDE on one interval between two events:
   * the parameter row passed to the user's function, the infusion rates
   * added to its result, and the real and integer data.
   *
   * Every quantity that can be a parameter carries its derivatives with
   * respect to the P global parameters of the schedule (the distinct
   * autodiff variables among all inputs, see <code>PMXDdeParameters</code>).
   */
  struct PMXDdeRhs {
    /// values of the parameter row theta
    std::vector<double> theta;
    /// for each entry of theta, the index of the global parameter it is, or -1
    std::vector<int> theta_index;
    /// infusion rate into each compartment
    Eigen::VectorXd rate;
    /// derivatives of the rates (nCmt x P), or empty if they are all zero
    Eigen::MatrixXd rate_sensitivities;
    /// real data passed to the right hand side
    const std::vector<double>* x_r = nullptr;
    /// integer data passed to the right hand side
    const std::vector<int>* x_i = nullptr;

    /**
     * Whether two right hand sides are the same function, so that no
     * discontinuity of the derivative separates the intervals they belong to.
     */
    bool operator==(const PMXDdeRhs& other) const {
      return theta == other.theta && theta_index == other.theta_index
             && rate == other.rate
             && rate_sensitivities.rows() == other.rate_sensitivities.rows()
             && rate_sensitivities.cols() == other.rate_sensitivities.cols()
             && rate_sensitivities == other.rate_sensitivities
             && *x_r == *other.x_r && *x_i == *other.x_i;
    }

    bool operator!=(const PMXDdeRhs& other) const { return !(*this == other); }
  };

  /**
   * The coupled forward sensitivity system of a PMX DDE, the system
   * integrated by <code>stan::math::internal::dde_retard_integrator</code>.
   *
   * The user's right hand side is
   *   y'(t) = f(t, y(t), y_lag(t), theta, x_r, x_i) + rate,
   * where column j of the nCmt x D matrix y_lag is y(t - tau_j). The
   * parameter row theta, the rates and the data are those of the current
   * interval (<code>set_rhs</code>).
   *
   * With sensitivities the state is z = (y, S_1, ..., S_P), where S_p is the
   * derivative of y with respect to the global parameter p, and
   *
   *   S_p' = J_y S_p + sum_j J_lag,j [S_p(t - tau_j) - y'(t - tau_j) dtau_j/dp]
   *          + sum_{k : theta_k is p} df/dtheta_k + drate/dp.
   *
   * Unlike Stan's <code>coupled_dde_system</code>, whose parameters are the
   * arguments of one call, the parameters here are those of the whole
   * schedule: the rows of theta of all events, the delays, the history, the
   * amounts, rates, times, intervals, bioavailabilities and lag times. A
   * parameter that the current interval does not depend on has zero
   * forcing in the sensitivity equation, but its sensitivity still evolves,
   * since earlier intervals made y depend on it.
   *
   * The Jacobian-vector products use nested reverse mode, one sweep per
   * compartment.
   *
   * @tparam F type of the user's right hand side, called as
   *   <code>f(t, y, y_lag, theta, x_r, x_i, msgs)</code> with an Eigen column
   *   vector y, an Eigen matrix y_lag and a std::vector theta.
   */
  template <typename F>
  class PMXDdeSystem {
  public:
    /**
     * @param f the user's right hand side
     * @param ncmt number of compartments
     * @param num_params number of global parameters P
     * @param delay_sensitivities P x D derivatives of the delays with
     *   respect to the global parameters
     * @param sensitivities whether to integrate the sensitivities; without
     *   them the system is the base system of nCmt components
     * @param msgs stream for messages of the right hand side
     */
    PMXDdeSystem(const F& f, int ncmt, size_t num_params,
                 const Eigen::MatrixXd& delay_sensitivities,
                 bool sensitivities, std::ostream* msgs)
      : f_(f),
        n_(ncmt),
        p_(sensitivities ? num_params : 0),
        d_(delay_sensitivities.cols()),
        delay_sensitivities_(sensitivities ? delay_sensitivities
                                           : Eigen::MatrixXd(0, d_)),
        lag_derivatives_(sensitivities && !delay_sensitivities.isZero()),
        msgs_(msgs),
        rhs_(nullptr) {}

    /**
     * Use the given right hand side from now on. The system keeps a
     * reference to it.
     */
    void set_rhs(const PMXDdeRhs& rhs) { rhs_ = &rhs; }

    /**
     * The current right hand side.
     */
    const PMXDdeRhs& rhs() const { return *rhs_; }

    size_t size() const { return n_ * (1 + p_); }

    size_t num_states() const { return n_; }

    size_t num_lag_derivatives() const { return lag_derivatives_ ? n_ : 0; }

    Eigen::MatrixXd delay_sensitivities() const { return delay_sensitivities_; }

    /**
     * The derivative of the base state, f + rate, at time t, state y and
     * lagged states y_lag.
     */
    Eigen::VectorXd base(double t, const Eigen::VectorXd& y,
                         const Eigen::MatrixXd& y_lag) const {
      const PMXDdeRhs& rhs = *rhs_;
      Eigen::VectorXd dy_dt = f_(t, y, y_lag, rhs.theta, *rhs.x_r, *rhs.x_i, msgs_);
      stan::math::check_size_match("pmx_solve_dde_rk45", "dy_dt", dy_dt.size(),
                                   "nCmt", n_);
      return dy_dt + rhs.rate;
    }

    /**
     * The residual of an equilibrium x of the current right hand side, whose
     * history is constant: f(t, x, [x, ..., x]) + rate.
     */
    template <typename T>
    Eigen::Matrix<T, -1, 1> equilibrium_residual(double t,
                                                 const Eigen::Matrix<T, -1, 1>& x) const {
      const PMXDdeRhs& rhs = *rhs_;
      const Eigen::Matrix<T, -1, -1> x_lag = x.replicate(1, d_);
      Eigen::Matrix<T, -1, 1> res = f_(t, x, x_lag, rhs.theta, *rhs.x_r, *rhs.x_i, msgs_);
      stan::math::check_size_match("pmx_solve_dde_rk45", "dy_dt", res.size(),
                                   "nCmt", n_);
      return res + rhs.rate;
    }

    /**
     * Derivatives of the equilibrium residual at x with respect to x
     * (nCmt x nCmt) and to the global parameters (nCmt x P).
     */
    void equilibrium_jacobian(double t, const Eigen::VectorXd& x,
                              Eigen::MatrixXd& jac_x, Eigen::MatrixXd& jac_p) const {
      using stan::math::var;
      const PMXDdeRhs& rhs = *rhs_;
      stan::math::nested_rev_autodiff nested;
      Eigen::Matrix<var, -1, 1> y(n_);
      for (size_t i = 0; i < n_; ++i) {
        y.coeffRef(i) = x.coeff(i);
      }
      // the same variables as y: the history of an equilibrium is constant
      const Eigen::Matrix<var, -1, -1> y_lag = y.replicate(1, d_);
      std::vector<var> theta(rhs.theta.begin(), rhs.theta.end());
      Eigen::Matrix<var, -1, 1> fy = f_(t, y, y_lag, theta, *rhs.x_r, *rhs.x_i, msgs_);
      stan::math::check_size_match("pmx_solve_dde_rk45", "dy_dt", fy.size(),
                                   "nCmt", n_);
      jac_x.resize(n_, n_);
      jac_p = Eigen::MatrixXd::Zero(n_, p_);
      for (size_t i = 0; i < n_; ++i) {
        fy.coeffRef(i).grad();
        for (size_t j = 0; j < n_; ++j) {
          jac_x.coeffRef(i, j) = y.coeff(j).adj();
        }
        for (size_t k = 0; k < theta.size(); ++k) {
          if (rhs.theta_index[k] >= 0) {
            jac_p.coeffRef(i, rhs.theta_index[k]) += theta[k].adj();
          }
        }
        if (rhs.rate_sensitivities.size() > 0) {
          jac_p.row(i) += rhs.rate_sensitivities.row(i);
        }
        nested.set_zero_all_adjoints();
      }
    }

    /**
     * The right hand side of the coupled system, in the form
     * <code>dde_retard_integrator</code> calls it.
     */
    void operator()(double t, const Eigen::VectorXd& z,
                    const Eigen::MatrixXd& z_lag,
                    const Eigen::MatrixXd& z_lag_deriv,
                    Eigen::VectorXd& dz_dt) const {
      using stan::math::var;
      dz_dt.resize(size());
      if (p_ == 0) {
        dz_dt = base(t, z, z_lag);
        return;
      }

      const PMXDdeRhs& rhs = *rhs_;
      stan::math::nested_rev_autodiff nested;

      Eigen::Matrix<var, -1, 1> y(n_);
      for (size_t i = 0; i < n_; ++i) {
        y.coeffRef(i) = z.coeff(i);
      }
      Eigen::Matrix<var, -1, -1> y_lag(n_, d_);
      for (size_t j = 0; j < d_; ++j) {
        for (size_t i = 0; i < n_; ++i) {
          y_lag.coeffRef(i, j) = z_lag.coeff(i, j);
        }
      }
      std::vector<var> theta(rhs.theta.begin(), rhs.theta.end());

      Eigen::Matrix<var, -1, 1> fy = f_(t, y, y_lag, theta, *rhs.x_r, *rhs.x_i, msgs_);
      stan::math::check_size_match("pmx_solve_dde_rk45", "dy_dt", fy.size(),
                                   "nCmt", n_);

      const Eigen::Map<const Eigen::MatrixXd> sens(z.data() + n_, n_, p_);
      Eigen::VectorXd ds(p_);
      for (size_t i = 0; i < n_; ++i) {
        dz_dt.coeffRef(i) = fy.coeff(i).val() + rhs.rate.coeff(i);
        fy.coeffRef(i).grad();

        // materialized: the adj() views do not support Eigen's direct
        // access, which matrix products may use
        const Eigen::VectorXd y_adj = y.adj();
        const Eigen::MatrixXd y_lag_adj = y_lag.adj();
        ds.noalias() = sens.transpose() * y_adj;
        for (size_t j = 0; j < d_; ++j) {
          const Eigen::Map<const Eigen::MatrixXd> sens_lag(z_lag.col(j).data() + n_, n_, p_);
          ds.noalias() += sens_lag.transpose() * y_lag_adj.col(j);
          if (lag_derivatives_) {
            ds -= delay_sensitivities_.col(j) * z_lag_deriv.col(j).dot(y_lag_adj.col(j));
          }
        }
        for (size_t k = 0; k < theta.size(); ++k) {
          if (rhs.theta_index[k] >= 0) {
            ds.coeffRef(rhs.theta_index[k]) += theta[k].adj();
          }
        }
        if (rhs.rate_sensitivities.size() > 0) {
          ds += rhs.rate_sensitivities.row(i).transpose();
        }
        for (size_t p = 0; p < p_; ++p) {
          dz_dt.coeffRef(n_ + n_ * p + i) = ds.coeff(p);
        }
        nested.set_zero_all_adjoints();
      }
    }

  private:
    const F& f_;
    const size_t n_;
    const size_t p_;
    const size_t d_;
    const Eigen::MatrixXd delay_sensitivities_;
    const bool lag_derivatives_;
    std::ostream* msgs_;
    const PMXDdeRhs* rhs_;
  };

}
}

#endif
