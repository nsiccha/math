#include <stan/math/rev.hpp>
#include <test/unit/math/rev/util.hpp>
#include <test/unit/math/rev/prob/expect_eq_diffs.hpp>
#include <gtest/gtest.h>
#include <string>
#include <vector>

template <typename T_prob>
inline void expect_propto_multinomial(std::vector<int>& ns1, T_prob theta1,
                                      std::vector<int>& ns2, T_prob theta2,
                                      std::string message) {
  expect_eq_diffs(stan::math::multinomial_lpmf<false>(ns1, theta1),
                  stan::math::multinomial_lpmf<false>(ns2, theta2),
                  stan::math::multinomial_lpmf<true>(ns1, theta1),
                  stan::math::multinomial_lpmf<true>(ns2, theta2), message);
}

TEST_F(AgradRev, DistributionsMultinomial_Propto) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::var;
  std::vector<int> ns;
  ns.push_back(1);
  ns.push_back(2);
  ns.push_back(3);
  Matrix<var, Dynamic, 1> theta1(3, 1);
  theta1 << 0.3, 0.5, 0.2;
  Matrix<var, Dynamic, 1> theta2(3, 1);
  theta2 << 0.1, 0.2, 0.7;

  expect_propto_multinomial(ns, theta1, ns, theta2, "var: theta");
}

TEST_F(AgradRev, DistributionsMultinomial_check_varis_on_stack) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::var;
  std::vector<int> ns;
  ns.push_back(1);
  ns.push_back(2);
  ns.push_back(3);
  Matrix<var, Dynamic, 1> theta(3, 1);
  theta << 0.3, 0.5, 0.2;

  test::check_varis_on_stack(stan::math::multinomial_lpmf<false>(ns, theta));
  test::check_varis_on_stack(stan::math::multinomial_lpmf<true>(ns, theta));
}

// A category with a zero count does not enter the density, so a zero
// probability there must get a zero gradient, not 0 * 0 / 0 = NaN.
TEST_F(AgradRev, DistributionsMultinomial_zero_count_zero_prob) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::var;
  Matrix<var, Dynamic, 1> theta(3);
  theta << 0.0, 0.6, 0.4;
  std::vector<int> ns{0, 2, 3};

  var lp = stan::math::multinomial_lpmf<false>(ns, theta);
  lp.grad();
  EXPECT_FLOAT_EQ(std::lgamma(6.0) - std::lgamma(3.0) - std::lgamma(4.0)
                      + 2 * std::log(0.6) + 3 * std::log(0.4),
                  lp.val());
  EXPECT_EQ(0.0, theta(0).adj());
  EXPECT_FLOAT_EQ(2 / 0.6, theta(1).adj());
  EXPECT_FLOAT_EQ(3 / 0.4, theta(2).adj());
}
