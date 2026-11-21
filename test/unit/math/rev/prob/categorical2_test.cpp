#include <stan/math/rev.hpp>
#include <test/unit/math/rev/util.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/prob/expect_eq_diffs.hpp>
#include <cmath>
#include <string>
#include <vector>

template <typename T_prob>
inline void expect_propto_categorical_lpmf(unsigned int n1, T_prob theta1,
                                           unsigned int n2, T_prob theta2,
                                           std::string message) {
  expect_eq_diffs(stan::math::categorical_lpmf<false>(n1, theta1),
                  stan::math::categorical_lpmf<false>(n2, theta2),
                  stan::math::categorical_lpmf<true>(n1, theta1),
                  stan::math::categorical_lpmf<true>(n2, theta2), message);
}

TEST_F(AgradRev, DistributionsCategorical_Propto) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::var;
  unsigned int n;
  Matrix<var, Dynamic, 1> theta1(3, 1);
  theta1 << 0.3, 0.5, 0.2;
  Matrix<var, Dynamic, 1> theta2(3, 1);
  theta2 << 0.1, 0.2, 0.7;

  n = 1;
  expect_propto_categorical_lpmf(n, theta1, n, theta2, "var: theta");

  n = 2;
  expect_propto_categorical_lpmf(n, theta1, n, theta2, "var: theta");

  n = 3;
  expect_propto_categorical_lpmf(n, theta1, n, theta2, "var: theta");
}

// The density of an array of outcomes reads only the observed entries of
// theta, so an unobserved entry that is exactly zero must get a zero
// gradient, not the 0 * inf = NaN of log(0)'s reverse pass.
TEST_F(AgradRev, DistributionsCategorical_vector_unobserved_zero_prob) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::var;
  Matrix<var, Dynamic, 1> theta(3);
  theta << 0.0, 0.6, 0.4;
  std::vector<int> ns{2, 3, 2};

  var lp = stan::math::categorical_lpmf<false>(ns, theta);
  lp.grad();
  EXPECT_FLOAT_EQ(2 * std::log(0.6) + std::log(0.4), lp.val());
  EXPECT_EQ(0.0, theta(0).adj());
  EXPECT_FLOAT_EQ(2 / 0.6, theta(1).adj());
  EXPECT_FLOAT_EQ(1 / 0.4, theta(2).adj());
}

// An unobserved entry that underflows to an exact zero but still depends on
// a parameter s: the gradient with respect to s is that of the observed
// entry alone, for the array overload as for the scalar one.
TEST_F(AgradRev, DistributionsCategorical_vector_matches_scalar_underflow) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::var;
  for (bool vectorised : {true, false}) {
    var s = 0.3;
    var a = stan::math::exp(s - 2000);
    var p = stan::math::inv_logit(s);
    Matrix<var, Dynamic, 1> theta(3);
    theta << a, p, 1 - p - a;
    ASSERT_EQ(0.0, theta(0).val());
    var lp;
    if (vectorised) {
      lp = stan::math::categorical_lpmf<false>(std::vector<int>{2}, theta);
    } else {
      lp = stan::math::categorical_lpmf<false>(2, theta);
    }
    lp.grad();
    EXPECT_FLOAT_EQ(std::log(p.val()), lp.val());
    EXPECT_FLOAT_EQ(1 - p.val(), s.adj()) << "vectorised " << vectorised;
    stan::math::recover_memory();
  }
}

// The array overload has the same value and gradient as the sum of the
// scalar overload over the outcomes, including an observed zero entry.
TEST_F(AgradRev, DistributionsCategorical_vector_matches_scalar_sum) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::var;
  Eigen::VectorXd theta_val(4);
  theta_val << 0.1, 0.0, 0.25, 0.65;
  std::vector<std::vector<int>> cases{{1, 4, 4, 3, 1, 4}, {3}, {2, 4}};
  for (const auto& ns : cases) {
    Matrix<var, Dynamic, 1> theta1 = theta_val;
    var lp1 = stan::math::categorical_lpmf<false>(ns, theta1);
    lp1.grad();
    double lp1_val = lp1.val();
    Eigen::VectorXd grad1 = theta1.adj();
    stan::math::recover_memory();

    Matrix<var, Dynamic, 1> theta2 = theta_val;
    var lp2 = 0;
    for (int n : ns) {
      lp2 += stan::math::categorical_lpmf<false>(n, theta2);
    }
    lp2.grad();
    EXPECT_FLOAT_EQ(lp2.val(), lp1_val);
    for (int k = 0; k < 4; ++k) {
      EXPECT_FLOAT_EQ(theta2(k).adj(), grad1(k)) << "k = " << k;
    }
    stan::math::recover_memory();
  }
}

TEST_F(AgradRev, DistributionsCategorical_vector_check_varis_on_stack) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::var;
  Matrix<var, Dynamic, 1> theta(3);
  theta << 0.3, 0.5, 0.2;
  std::vector<int> ns{1, 3, 3};
  test::check_varis_on_stack(stan::math::categorical_lpmf<false>(ns, theta));
  test::check_varis_on_stack(stan::math::categorical_lpmf<true>(ns, theta));
}
