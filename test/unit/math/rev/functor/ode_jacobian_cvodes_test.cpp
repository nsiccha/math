#include <stan/math/rev.hpp>
#include <test/unit/math/rev/util.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace {

using stan::math::var;

/**
 * Right-hand side whose variadic arguments mix every kind the Jacobian
 * convention has to order: a parameter vector, a real array (only its first
 * element is used), an integer array and a real scalar.
 *
 * y0' = -theta0 y0 + k sin(t)
 * y1' = (theta0 + 0.01 x_i0) y0 - theta1 x_r0 y1^2
 * y2' = theta1 x_r0 y1^2 - k y2
 */
struct mixed_rhs {
  template <typename T_t, typename T_y, typename T_theta, typename T_k>
  Eigen::Matrix<stan::return_type_t<T_t, T_y, T_theta, T_k>, Eigen::Dynamic, 1>
  operator()(const T_t& t, const T_y& y, std::ostream* msgs,
             const T_theta& theta, const std::vector<double>& x_r,
             const std::vector<int>& x_i, const T_k& k) const {
    using stan::math::sin;
    Eigen::Matrix<stan::return_type_t<T_t, T_y, T_theta, T_k>, Eigen::Dynamic,
                  1>
        dy(3);
    dy(0) = -theta(0) * y(0) + k * sin(t);
    dy(1) = (theta(0) + 0.01 * x_i[0]) * y(0) - theta(1) * x_r[0] * y(1) * y(1);
    dy(2) = theta(1) * x_r[0] * y(1) * y(1) - k * y(2);
    return dy;
  }
};

/**
 * Jacobian of mixed_rhs: columns y0, y1, y2, theta0, theta1, x_r0, x_r1, k
 * (the integer array has no columns).
 */
struct mixed_jac {
  template <typename T_t, typename T_y, typename T_theta, typename T_k>
  Eigen::MatrixXd operator()(const T_t& t, const T_y& y, std::ostream* msgs,
                             const T_theta& theta,
                             const std::vector<double>& x_r,
                             const std::vector<int>& x_i, const T_k& k) const {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, 8);
    J(0, 0) = -theta(0);
    J(0, 3) = -y(0);
    J(0, 7) = std::sin(t);
    J(1, 0) = theta(0) + 0.01 * x_i[0];
    J(1, 1) = -2 * theta(1) * x_r[0] * y(1);
    J(1, 3) = y(0);
    J(1, 4) = -x_r[0] * y(1) * y(1);
    J(1, 5) = -theta(1) * y(1) * y(1);
    J(2, 1) = 2 * theta(1) * x_r[0] * y(1);
    J(2, 2) = -k;
    J(2, 4) = x_r[0] * y(1) * y(1);
    J(2, 5) = theta(1) * y(1) * y(1);
    J(2, 7) = -y(2);
    return J;
  }
};

/**
 * A Jacobian that is right with respect to the state but zero with
 * respect to the arguments, to show the solver uses the user Jacobian.
 */
struct state_only_jac {
  template <typename... Args>
  Eigen::MatrixXd operator()(Args&&... args) const {
    Eigen::MatrixXd J = mixed_jac()(args...);
    J.rightCols(5).setZero();
    return J;
  }
};

/** A Jacobian with a column missing. */
struct short_jac {
  template <typename... Args>
  Eigen::MatrixXd operator()(Args&&... args) const {
    return mixed_jac()(args...).leftCols(7);
  }
};

/** A Jacobian with a row missing. */
struct short_rows_jac {
  template <typename... Args>
  Eigen::MatrixXd operator()(Args&&... args) const {
    return mixed_jac()(args...).topRows(2);
  }
};

enum class solver { rk45, ckrk, adams, bdf, adjoint };

const std::vector<solver> all_solvers{solver::rk45, solver::ckrk, solver::adams,
                                      solver::bdf};

std::string solver_name(solver s) {
  switch (s) {
    case solver::rk45:
      return "rk45";
    case solver::ckrk:
      return "ckrk";
    case solver::adams:
      return "adams";
    case solver::bdf:
      return "bdf";
    default:
      return "adjoint";
  }
}

/**
 * Solve with the autodiff path (UseJac false) or the user Jacobian path
 * (UseJac true) of the given solver.
 */
template <bool UseJac, typename FJac = mixed_jac, typename T_y0, typename T_t0,
          typename T_ts, typename T_theta, typename T_k>
auto solve(solver s, const T_y0& y0, const T_t0& t0,
           const std::vector<T_ts>& ts, const T_theta& theta,
           const std::vector<double>& x_r, const std::vector<int>& x_i,
           const T_k& k) {
  const mixed_rhs f;
  const FJac jac;
  const double rtol = 1e-10;
  const double atol = 1e-10;
  const long int max_steps = 1e8;  // NOLINT(runtime/int)
  switch (s) {
    case solver::rk45:
      if constexpr (UseJac) {
        return stan::math::ode_rk45_tol(f, jac, y0, t0, ts, rtol, atol,
                                        max_steps, nullptr, theta, x_r, x_i, k);
      } else {
        return stan::math::ode_rk45_tol(f, y0, t0, ts, rtol, atol, max_steps,
                                        nullptr, theta, x_r, x_i, k);
      }
    case solver::ckrk:
      if constexpr (UseJac) {
        return stan::math::ode_ckrk_tol(f, jac, y0, t0, ts, rtol, atol,
                                        max_steps, nullptr, theta, x_r, x_i, k);
      } else {
        return stan::math::ode_ckrk_tol(f, y0, t0, ts, rtol, atol, max_steps,
                                        nullptr, theta, x_r, x_i, k);
      }
    case solver::adams:
      if constexpr (UseJac) {
        return stan::math::ode_adams_tol(f, jac, y0, t0, ts, rtol, atol,
                                         max_steps, nullptr, theta, x_r, x_i,
                                         k);
      } else {
        return stan::math::ode_adams_tol(f, y0, t0, ts, rtol, atol, max_steps,
                                         nullptr, theta, x_r, x_i, k);
      }
    case solver::bdf:
      if constexpr (UseJac) {
        return stan::math::ode_bdf_tol(f, jac, y0, t0, ts, rtol, atol,
                                       max_steps, nullptr, theta, x_r, x_i, k);
      } else {
        return stan::math::ode_bdf_tol(f, y0, t0, ts, rtol, atol, max_steps,
                                       nullptr, theta, x_r, x_i, k);
      }
    default: {
      const Eigen::VectorXd atol_vec = Eigen::VectorXd::Constant(3, atol);
      const int checkpoints = 150;
      const int hermite = 1;
      const int bdf = 2;
      if constexpr (UseJac) {
        return stan::math::ode_adjoint_tol_ctl(
            f, jac, y0, t0, ts, rtol, atol_vec, rtol, atol_vec, rtol, atol,
            max_steps, checkpoints, hermite, bdf, bdf, nullptr, theta, x_r, x_i,
            k);
      } else {
        return stan::math::ode_adjoint_tol_ctl(
            f, y0, t0, ts, rtol, atol_vec, rtol, atol_vec, rtol, atol,
            max_steps, checkpoints, hermite, bdf, bdf, nullptr, theta, x_r, x_i,
            k);
      }
    }
  }
}

const std::vector<double> x_r{1.5, -7.0};
const std::vector<int> x_i{3};
const std::vector<double> ts_d{0.5, 1.0, 2.5, 4.0};

Eigen::VectorXd y0_d() {
  Eigen::VectorXd y0(3);
  y0 << 1.0, 0.5, 0.25;
  return y0;
}

Eigen::VectorXd theta_d() {
  Eigen::VectorXd theta(2);
  theta << 0.8, 0.6;
  return theta;
}

const double k_d = 0.4;
const double t0_d = 0.1;

/**
 * Inputs of one solve as autodiff variables, in a fixed order: y0, theta,
 * k, t0, ts. Which of them are autodiff is chosen by the flags; data inputs
 * enter the solve as doubles.
 */
struct var_inputs {
  Eigen::Matrix<var, Eigen::Dynamic, 1> y0;
  Eigen::Matrix<var, Eigen::Dynamic, 1> theta;
  var k;
  var t0;
  std::vector<var> ts;
  var_inputs()
      : y0(y0_d().cast<var>()),
        theta(theta_d().cast<var>()),
        k(k_d),
        t0(t0_d),
        ts(ts_d.begin(), ts_d.end()) {}
  std::vector<var> all() const {
    std::vector<var> v(y0.data(), y0.data() + y0.size());
    v.insert(v.end(), theta.data(), theta.data() + theta.size());
    v.push_back(k);
    v.push_back(t0);
    v.insert(v.end(), ts.begin(), ts.end());
    return v;
  }
};

/**
 * Value and gradient with respect to every input of every output of one
 * solve.
 */
using outputs = std::vector<std::pair<double, std::vector<double>>>;

template <bool UseJac, typename FJac = mixed_jac>
outputs solve_outputs(solver s, bool y0_var, bool theta_var, bool k_var,
                      bool t0_var, bool ts_var) {
  outputs out;
  var_inputs in;
  auto run = [&](const auto& y0, const auto& theta, const auto& k,
                 const auto& t0, const auto& ts) {
    auto ys = solve<UseJac, FJac>(s, y0, t0, ts, theta, x_r, x_i, k);
    const std::vector<var> inputs = in.all();
    for (const auto& y : ys) {
      for (Eigen::Index n = 0; n < y.size(); ++n) {
        if constexpr (stan::is_var<stan::value_type_t<
                          std::decay_t<decltype(y)>>>::value) {
          stan::math::set_zero_all_adjoints();
          stan::math::grad(y(n).vi_);
          std::vector<double> g;
          for (const var& x : inputs) {
            g.push_back(x.adj());
          }
          out.emplace_back(y(n).val(), g);
        } else {
          out.emplace_back(y(n), std::vector<double>(inputs.size(), 0.0));
        }
      }
    }
  };
  // dispatch the five flags to autodiff or double arguments
  auto with_ts
      = [&](const auto& y0, const auto& theta, const auto& k, const auto& t0) {
          if (ts_var) {
            run(y0, theta, k, t0, in.ts);
          } else {
            run(y0, theta, k, t0, ts_d);
          }
        };
  auto with_t0 = [&](const auto& y0, const auto& theta, const auto& k) {
    if (t0_var) {
      with_ts(y0, theta, k, in.t0);
    } else {
      with_ts(y0, theta, k, t0_d);
    }
  };
  auto with_k = [&](const auto& y0, const auto& theta) {
    if (k_var) {
      with_t0(y0, theta, in.k);
    } else {
      with_t0(y0, theta, k_d);
    }
  };
  auto with_theta = [&](const auto& y0) {
    if (theta_var) {
      with_k(y0, in.theta);
    } else {
      with_k(y0, theta_d());
    }
  };
  if (y0_var) {
    with_theta(in.y0);
  } else {
    with_theta(y0_d());
  }
  stan::math::recover_memory();
  return out;
}

void expect_outputs_near(const outputs& expected, const outputs& actual,
                         double tol, const std::string& context) {
  ASSERT_EQ(expected.size(), actual.size()) << context;
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(expected[i].first, actual[i].first,
                tol * std::max(1.0, std::fabs(expected[i].first)))
        << context << ", output " << i;
    ASSERT_EQ(expected[i].second.size(), actual[i].second.size());
    for (size_t j = 0; j < expected[i].second.size(); ++j) {
      EXPECT_NEAR(expected[i].second[j], actual[i].second[j],
                  tol * std::max(1.0, std::fabs(expected[i].second[j])))
          << context << ", output " << i << ", input " << j;
    }
  }
}

}  // namespace

/**
 * Every solver's user-Jacobian variant reproduces the values and the
 * gradients of its autodiff path for every combination of autodiff and
 * data inputs.
 */
TEST(ode_jacobian, matches_autodiff_path) {
  for (solver s : all_solvers) {
    for (int mask = 0; mask < 32; ++mask) {
      const bool y0_var = mask & 1;
      const bool theta_var = mask & 2;
      const bool k_var = mask & 4;
      const bool t0_var = mask & 8;
      const bool ts_var = mask & 16;
      const std::string context
          = solver_name(s) + ", mask " + std::to_string(mask);
      const outputs ad
          = solve_outputs<false>(s, y0_var, theta_var, k_var, t0_var, ts_var);
      const outputs jac
          = solve_outputs<true>(s, y0_var, theta_var, k_var, t0_var, ts_var);
      expect_outputs_near(ad, jac, 1e-7, context);
    }
  }
}

/**
 * The solvers use the user Jacobian: one that is zero with respect to the
 * arguments gives zero gradients with respect to theta and k, while the
 * gradients with respect to y0 are unaffected.
 */
TEST(ode_jacobian, uses_user_jacobian) {
  for (solver s : all_solvers) {
    const outputs ad = solve_outputs<false>(s, true, true, true, false, false);
    const outputs state_only = solve_outputs<true, state_only_jac>(
        s, true, true, true, false, false);
    ASSERT_EQ(ad.size(), state_only.size());
    for (size_t i = 0; i < ad.size(); ++i) {
      EXPECT_NEAR(ad[i].first, state_only[i].first, 1e-7);
      for (size_t j = 0; j < 3; ++j) {
        EXPECT_NEAR(ad[i].second[j], state_only[i].second[j], 1e-7)
            << solver_name(s) << ", output " << i << ", y0 " << j;
      }
      for (size_t j = 3; j < 6; ++j) {
        EXPECT_EQ(state_only[i].second[j], 0.0)
            << solver_name(s) << ", output " << i << ", argument " << j;
      }
    }
  }
}

/**
 * A Jacobian of the wrong size is an error, not a silent misreading.
 */
TEST(ode_jacobian, wrong_size_throws) {
  for (solver s : all_solvers) {
    EXPECT_THROW(
        (solve_outputs<true, short_jac>(s, false, true, false, false, false)),
        std::invalid_argument)
        << solver_name(s);
    EXPECT_THROW((solve_outputs<true, short_rows_jac>(s, false, true, false,
                                                      false, false)),
                 std::invalid_argument)
        << solver_name(s);
  }
  stan::math::recover_memory();
}

/**
 * With only data inputs no sensitivities are computed; the solution equals
 * the plain solver's.
 */
TEST(ode_jacobian, all_data) {
  for (solver s : all_solvers) {
    const auto plain
        = solve<false>(s, y0_d(), t0_d, ts_d, theta_d(), x_r, x_i, k_d);
    const auto with_jac
        = solve<true>(s, y0_d(), t0_d, ts_d, theta_d(), x_r, x_i, k_d);
    ASSERT_EQ(plain.size(), with_jac.size());
    for (size_t i = 0; i < plain.size(); ++i) {
      EXPECT_MATRIX_FLOAT_EQ(plain[i], with_jac[i]);
    }
  }
}

/**
 * The parameter block's columns follow count_vars order across scalars,
 * Eigen types (column-major), arrays and tuples; integers get no columns.
 */
TEST(ode_jacobian, var_columns) {
  using stan::math::internal::ode_jacobian_var_columns;
  Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic> m(2, 2);
  m << 1, 2, 3, 4;
  const std::vector<double> data{1, 2, 3};
  const std::vector<int> ints{1, 2};
  const std::vector<var> vs{5, 6};
  const auto tup
      = std::make_tuple(var(7), 8.0, std::vector<int>{1}, std::vector<var>{9});
  const int three = 3;
  const double two_five = 2.5;
  std::size_t num_columns = 99;
  const std::vector<std::size_t> columns = ode_jacobian_var_columns(
      num_columns, data, m, ints, three, vs, tup, two_five);
  // data: 0-2; m: 3-6; ints: none; 3: none; vs: 7-8;
  // tup: var 9, double 10, ints none, var 11; 2.5: 12
  EXPECT_EQ(num_columns, 13);
  EXPECT_EQ(columns, (std::vector<std::size_t>{3, 4, 5, 6, 7, 8, 9, 11}));
  EXPECT_EQ(columns.size(), stan::math::count_vars(data, m, ints, three, vs)
                                + stan::math::count_vars(tup)
                                + stan::math::count_vars(two_five));
  stan::math::recover_memory();
}
