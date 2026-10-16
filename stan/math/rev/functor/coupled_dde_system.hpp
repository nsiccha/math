#ifndef STAN_MATH_REV_FUNCTOR_COUPLED_DDE_SYSTEM_HPP
#define STAN_MATH_REV_FUNCTOR_COUPLED_DDE_SYSTEM_HPP

#include <stan/math/prim/functor/coupled_dde_system.hpp>
#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/fun/value_of.hpp>
#include <stan/math/prim/functor/apply.hpp>
#include <stan/math/prim/functor/for_each.hpp>
#include <stan/math/prim/err.hpp>
#include <ostream>
#include <tuple>

namespace stan {
namespace math {

/**
 * The coupled DDE system when the initial state, the delays or the extra
 * arguments contain autodiff variables.
 *
 * <p>Let P be the number of parameters theta: the autodiff variables of the
 * initial state y0, then those of the extra arguments, then those of the
 * delays tau (the order in which <code>ode_store_sensitivities</code> reads
 * them). The coupled system has N + N * P components: the base state y,
 * followed by the sensitivities S_p = dy / dtheta_p, p = 1, ..., P, each of
 * size N.
 *
 * <p>Differentiating y'(t) = f(t, y(t), y(t - tau_1), ..., y(t - tau_D),
 * theta) gives
 *
 * \f[
 *   S_p'(t) = J_y S_p(t) + \sum_j J_{lag,j} \left[ S_p(t - \tau_j)
 *     - y'(t - \tau_j) \frac{\partial \tau_j}{\partial \theta_p} \right]
 *     + \frac{\partial f}{\partial \theta_p},
 * \f]
 *
 * so the sensitivities are further delayed components integrated through the
 * same history as y. Before t0 the history is y0, so S_p = dy0 / dtheta_p and
 * y' = 0 there. The integrator supplies y'(t - tau_j) from the derivative of
 * the dense output when tau contains autodiff variables.
 *
 * <p>The Jacobian-vector products are computed with nested reverse mode, one
 * sweep per component of f, as in <code>coupled_ode_system</code>. As there,
 * the extra arguments are deep copied onto the nochain part of the autodiff
 * tape on construction, and their adjoints are zeroed separately.
 *
 * @tparam F base DDE right hand side functor. Must provide
 *   <code>operator()(double t,
 *     const Eigen::Matrix<T_y, Eigen::Dynamic, 1>& y,
 *     const Eigen::Matrix<T_y, Eigen::Dynamic, Eigen::Dynamic>& y_lag,
 *     std::ostream* msgs, const T_args&... args)</code>
 *   returning the derivative vector.
 * @tparam T_y0 scalar type of the initial state
 * @tparam T_tau type of the delay vector
 * @tparam Args types of the extra arguments of F
 */
template <typename F, typename T_y0, typename T_tau, typename... Args>
struct coupled_dde_system_impl<false, F, T_y0, T_tau, Args...> {
  const F& f_;
  const Eigen::Matrix<T_y0, Eigen::Dynamic, 1>& y0_;
  std::tuple<decltype(deep_copy_vars(std::declval<const Args&>()))...>
      local_args_tuple_;
  const size_t num_y0_vars_;
  const size_t num_args_vars_;
  const size_t num_tau_vars_;
  const size_t N_;
  const size_t D_;
  Eigen::VectorXd args_adjoints_;
  Eigen::VectorXd y_adjoints_;
  Eigen::MatrixXd y_lag_adjoints_;
  std::ostream* msgs_;

  /**
   * @param[in] f the base DDE system functor
   * @param[in] y0 the initial state of the base DDE
   * @param[in] tau the delays
   * @param[in, out] msgs stream for messages
   * @param[in] args other additional arguments
   */
  coupled_dde_system_impl(const F& f,
                          const Eigen::Matrix<T_y0, Eigen::Dynamic, 1>& y0,
                          const T_tau& tau, std::ostream* msgs,
                          const Args&... args)
      : f_(f),
        y0_(y0),
        local_args_tuple_(deep_copy_vars(args)...),
        num_y0_vars_(count_vars(y0_)),
        num_args_vars_(count_vars(args...)),
        num_tau_vars_(count_vars(tau)),
        N_(y0.size()),
        D_(tau.size()),
        args_adjoints_(num_args_vars_),
        y_adjoints_(N_),
        y_lag_adjoints_(N_, D_),
        msgs_(msgs) {}

  /**
   * Calculates the right hand side of the coupled system.
   *
   * @param[in] t time
   * @param[in] z state of the coupled system; size <code>size()</code>
   * @param[in] z_lag lagged states of the coupled system; column j is
   *   z(t - tau_j)
   * @param[in] z_lag_deriv time derivatives of the lagged base states;
   *   column j is y'(t - tau_j). Only read when the delays are autodiff
   *   variables.
   * @param[out] dz_dt derivative of the coupled state; size
   *   <code>size()</code>
   * @throw exception if the base DDE function does not return the
   *    expected number of derivatives, N.
   */
  void operator()(double t, const Eigen::VectorXd& z,
                  const Eigen::MatrixXd& z_lag,
                  const Eigen::MatrixXd& z_lag_deriv, Eigen::VectorXd& dz_dt) {
    dz_dt.resize(size());

    // Run nested autodiff in this scope
    nested_rev_autodiff nested;

    Eigen::Matrix<var, Eigen::Dynamic, 1> y_vars(N_);
    for (size_t n = 0; n < N_; ++n) {
      y_vars.coeffRef(n) = z.coeff(n);
    }
    Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic> y_lag_vars(N_, D_);
    for (size_t j = 0; j < D_; ++j) {
      for (size_t n = 0; n < N_; ++n) {
        y_lag_vars.coeffRef(n, j) = z_lag.coeff(n, j);
      }
    }

    Eigen::Matrix<var, Eigen::Dynamic, 1> f_y_t_vars = math::apply(
        [&](auto&&... args) {
          return f_(t, y_vars, y_lag_vars, msgs_, args...);
        },
        local_args_tuple_);

    check_size_match("coupled_dde_system", "dy_dt", f_y_t_vars.size(), "states",
                     N_);

    const size_t num_params = num_y0_vars_ + num_args_vars_ + num_tau_vars_;
    const size_t first_tau_param = num_y0_vars_ + num_args_vars_;

    for (size_t i = 0; i < N_; ++i) {
      dz_dt.coeffRef(i) = f_y_t_vars.coeffRef(i).val();
      f_y_t_vars.coeffRef(i).grad();

      y_adjoints_ = y_vars.adj();
      y_lag_adjoints_ = y_lag_vars.adj();

      if (args_adjoints_.size() > 0) {
        args_adjoints_.setZero();
      }

      math::apply(
          [&](auto&&... args) {
            accumulate_adjoints(args_adjoints_.data(), args...);
          },
          local_args_tuple_);

      // The vars here do not live on the nested stack so must be zero'd
      // separately
      stan::math::for_each([](auto&& arg) { zero_adjoints(arg); },
                           local_args_tuple_);

      // No need to zero adjoints after last sweep
      if (i + 1 < N_) {
        nested.set_zero_all_adjoints();
      }

      for (size_t p = 0; p < num_params; ++p) {
        const size_t offset = N_ + N_ * p;
        double temp_deriv = z.segment(offset, N_).dot(y_adjoints_);
        for (size_t j = 0; j < D_; ++j) {
          temp_deriv
              += z_lag.col(j).segment(offset, N_).dot(y_lag_adjoints_.col(j));
        }
        if (p >= first_tau_param) {
          const size_t j = p - first_tau_param;
          temp_deriv -= z_lag_deriv.col(j).dot(y_lag_adjoints_.col(j));
        } else if (p >= num_y0_vars_) {
          temp_deriv += args_adjoints_.coeff(p - num_y0_vars_);
        }
        dz_dt.coeffRef(offset + i) = temp_deriv;
      }
    }
  }

  /**
   * Returns the size of the coupled system.
   */
  size_t size() const {
    return N_ + N_ * (num_y0_vars_ + num_args_vars_ + num_tau_vars_);
  }

  /**
   * Returns the number of states of the base system.
   */
  size_t num_states() const { return N_; }

  /**
   * Returns the P x D matrix of the derivatives of the delays with respect to
   * the sensitivity parameters: the identity in the rows of the delays if
   * they are autodiff variables, otherwise zero.
   */
  Eigen::MatrixXd delay_sensitivities() const {
    Eigen::MatrixXd dtau = Eigen::MatrixXd::Zero(
        num_y0_vars_ + num_args_vars_ + num_tau_vars_, D_);
    for (size_t j = 0; j < num_tau_vars_; ++j) {
      dtau(num_y0_vars_ + num_args_vars_ + j, j) = 1;
    }
    return dtau;
  }

  /**
   * Returns the number of leading components whose lagged derivatives the
   * right hand side needs: the base states if the delays are autodiff
   * variables, otherwise none.
   */
  size_t num_lag_derivatives() const { return num_tau_vars_ > 0 ? N_ : 0; }

  /**
   * Returns the initial state of the coupled system, which is also its
   * constant initial history: the initial state of the base system, the
   * identity for the sensitivities with respect to the initial state and
   * zero for the other sensitivities.
   */
  Eigen::VectorXd initial_state() const {
    Eigen::VectorXd initial = Eigen::VectorXd::Zero(size());
    for (size_t i = 0; i < N_; i++) {
      initial.coeffRef(i) = value_of(y0_.coeff(i));
    }
    for (size_t i = 0; i < num_y0_vars_; i++) {
      initial.coeffRef(N_ + i * N_ + i) = 1.0;
    }
    return initial;
  }
};

}  // namespace math
}  // namespace stan
#endif
