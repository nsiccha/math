#include <stan/math/rev.hpp>
#include <test/unit/math/rev/util.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

using stan::math::var;

/**
 * Robertson's chemical kinetics as a DAE; the variadic arguments are the
 * three rate constants, an unused real array and an unused integer array.
 */
struct chem_residual {
  template <typename T_yy, typename T_yp, typename T_p>
  Eigen::Matrix<stan::return_type_t<T_yy, T_yp, T_p>, Eigen::Dynamic, 1>
  operator()(double t, const T_yy& yy, const T_yp& yp, std::ostream* msgs,
             const std::vector<T_p>& p, const std::vector<double>& x_r,
             const std::vector<int>& x_i) const {
    Eigen::Matrix<stan::return_type_t<T_yy, T_yp, T_p>, Eigen::Dynamic, 1> r(3);
    r(0) = yp(0) + p[0] * yy(0) - p[1] * yy(1) * yy(2);
    r(1) = yp(1) - p[0] * yy(0) + p[1] * yy(1) * yy(2) + p[2] * yy(1) * yy(1);
    r(2) = yy(0) + yy(1) + yy(2) - 1.0;
    return r;
  }
};

/**
 * Jacobian of chem_residual: columns yy0-2, yp0-2, p0-2, x_r0-1 (the
 * integer array has no columns).
 */
struct chem_jacobian {
  template <typename T_yy, typename T_yp, typename T_p>
  Eigen::MatrixXd operator()(double t, const T_yy& yy, const T_yp& yp,
                             std::ostream* msgs, const std::vector<T_p>& p,
                             const std::vector<double>& x_r,
                             const std::vector<int>& x_i) const {
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, 11);
    J(0, 0) = p[0];
    J(0, 1) = -p[1] * yy(2);
    J(0, 2) = -p[1] * yy(1);
    J(0, 3) = 1;
    J(0, 6) = yy(0);
    J(0, 7) = -yy(1) * yy(2);
    J(1, 0) = -p[0];
    J(1, 1) = p[1] * yy(2) + 2 * p[2] * yy(1);
    J(1, 2) = p[1] * yy(1);
    J(1, 4) = 1;
    J(1, 6) = -yy(0);
    J(1, 7) = yy(1) * yy(2);
    J(1, 8) = yy(1) * yy(1);
    J(2, 0) = 1;
    J(2, 1) = 1;
    J(2, 2) = 1;
    return J;
  }
};

/** Right with respect to yy and yp, zero with respect to the arguments. */
struct chem_state_only_jacobian {
  template <typename... Args>
  Eigen::MatrixXd operator()(Args&&... args) const {
    Eigen::MatrixXd J = chem_jacobian()(args...);
    J.rightCols(5).setZero();
    return J;
  }
};

/** A Jacobian with a column missing. */
struct chem_short_jacobian {
  template <typename... Args>
  Eigen::MatrixXd operator()(Args&&... args) const {
    return chem_jacobian()(args...).leftCols(10);
  }
};

const std::vector<double> ts{1, 10, 100};
const std::vector<double> x_r{2.0, 3.0};
const std::vector<int> x_i{1};

/**
 * Values and gradients with respect to (yy0, yp0, p) of every output,
 * through the autodiff path (no Jacobian) or the user-Jacobian path.
 */
template <bool UseJac, typename FJac = chem_jacobian>
std::vector<std::pair<double, std::vector<double>>> solve(bool yy_var,
                                                          bool yp_var,
                                                          bool p_var) {
  Eigen::Matrix<var, Eigen::Dynamic, 1> yy0(3), yp0(3);
  yy0 << 1.0, 0.0, 0.0;
  yp0 << -0.04, 0.04, 0.0;
  std::vector<var> p{0.040, 1.0e4, 3.0e7};
  std::vector<var> inputs(yy0.data(), yy0.data() + 3);
  inputs.insert(inputs.end(), yp0.data(), yp0.data() + 3);
  inputs.insert(inputs.end(), p.begin(), p.end());
  std::vector<std::pair<double, std::vector<double>>> out;
  auto run = [&](const auto& yy, const auto& yp, const auto& pp) {
    const auto ys = [&]() {
      if constexpr (UseJac) {
        return stan::math::dae(chem_residual(), FJac(), yy, yp, 0.0, ts,
                               nullptr, pp, x_r, x_i);
      } else {
        return stan::math::dae(chem_residual(), yy, yp, 0.0, ts, nullptr, pp,
                               x_r, x_i);
      }
    }();
    if constexpr (stan::is_var<stan::value_type_t<
                      std::decay_t<decltype(ys[0])>>>::value) {
      for (const auto& y : ys) {
        for (int n = 0; n < 3; ++n) {
          stan::math::set_zero_all_adjoints();
          stan::math::grad(y(n).vi_);
          std::vector<double> g;
          for (const var& x : inputs) {
            g.push_back(x.adj());
          }
          out.emplace_back(y(n).val(), g);
        }
      }
    }
  };
  const Eigen::VectorXd yy_d = stan::math::value_of(yy0);
  const Eigen::VectorXd yp_d = stan::math::value_of(yp0);
  const std::vector<double> p_d = stan::math::value_of(p);
  auto with_p = [&](const auto& yy, const auto& yp) {
    if (p_var) {
      run(yy, yp, p);
    } else {
      run(yy, yp, p_d);
    }
  };
  auto with_yp = [&](const auto& yy) {
    if (yp_var) {
      with_p(yy, yp0);
    } else {
      with_p(yy, yp_d);
    }
  };
  if (yy_var) {
    with_yp(yy0);
  } else {
    with_yp(yy_d);
  }
  stan::math::recover_memory();
  return out;
}

}  // namespace

/**
 * The dae overloads taking a Jacobian reproduce the values and gradients
 * of the autodiff path for every combination of autodiff and data
 * initial states and parameters, up to the solver tolerance (the autodiff
 * path's Newton matrix is a finite-difference approximation, so the step
 * sequences differ slightly).
 */
TEST(dae_jacobian, matches_autodiff_path) {
  for (int mask = 1; mask < 8; ++mask) {
    const bool yy_var = mask & 1;
    const bool yp_var = mask & 2;
    const bool p_var = mask & 4;
    const auto ad = solve<false>(yy_var, yp_var, p_var);
    const auto jac = solve<true>(yy_var, yp_var, p_var);
    ASSERT_EQ(ad.size(), jac.size());
    for (size_t i = 0; i < ad.size(); ++i) {
      EXPECT_NEAR(ad[i].first, jac[i].first,
                  1e-6 * std::max(1.0, std::fabs(ad[i].first)))
          << "mask " << mask << ", output " << i;
      for (size_t j = 0; j < ad[i].second.size(); ++j) {
        EXPECT_NEAR(ad[i].second[j], jac[i].second[j],
                    1e-5 * std::max(1.0, std::fabs(ad[i].second[j])))
            << "mask " << mask << ", output " << i << ", input " << j;
      }
    }
  }
}

/**
 * The solver uses the user Jacobian: one that is zero with respect to the
 * arguments gives zero gradients with respect to the rate constants.
 */
TEST(dae_jacobian, uses_user_jacobian) {
  const auto state_only
      = solve<true, chem_state_only_jacobian>(true, false, true);
  const auto ad = solve<false>(true, false, true);
  for (size_t i = 0; i < ad.size(); ++i) {
    EXPECT_NEAR(ad[i].first, state_only[i].first,
                1e-6 * std::max(1.0, std::fabs(ad[i].first)));
    for (size_t j = 6; j < 9; ++j) {
      EXPECT_EQ(state_only[i].second[j], 0.0) << "output " << i;
    }
  }
}

/** A Jacobian of the wrong size is an error. */
TEST(dae_jacobian, wrong_size_throws) {
  EXPECT_THROW((solve<true, chem_short_jacobian>(false, false, true)),
               std::invalid_argument);
  stan::math::recover_memory();
}
