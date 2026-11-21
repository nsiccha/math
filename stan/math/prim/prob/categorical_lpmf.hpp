#ifndef STAN_MATH_PRIM_PROB_CATEGORICAL_LPMF_HPP
#define STAN_MATH_PRIM_PROB_CATEGORICAL_LPMF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/fun/to_ref.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/prim/functor/partials_propagator.hpp>
#include <cmath>
#include <vector>

namespace stan {
namespace math {

// Categorical(n|theta)  [0 < n <= N;   0 <= theta[n] <= 1;  SUM theta = 1]
template <bool propto, typename T_prob,
          require_eigen_col_vector_t<T_prob>* = nullptr>
inline return_type_t<T_prob> categorical_lpmf(int n, const T_prob& theta) {
  static constexpr const char* function = "categorical_lpmf";
  using std::log;

  check_bounded(function, "Number of categories", n, 1, theta.size());
  ref_type_t<T_prob> theta_ref = theta;
  check_simplex(function, "Probabilities parameter", value_of(theta_ref));

  if constexpr (include_summand<propto, T_prob>::value) {
    return log(theta_ref.coeff(n - 1));
  }
  return 0.0;
}

/** \ingroup multivar_dists
 * Categorical log PMF summed over an array of outcomes,
 * Categorical(ns | theta).
 *
 * The density reads only the observed entries of theta: an entry no
 * outcome names gets a zero partial, also where it is exactly zero.
 *
 * @param ns Outcomes, each in 1:size(theta)
 * @param theta Probabilities, a simplex
 * @return log probability of the outcomes
 * @throw std::domain_error if an outcome is not in 1:size(theta)
 * @throw std::domain_error if theta is not a simplex
 */
template <bool propto, typename T_prob,
          require_eigen_col_vector_t<T_prob>* = nullptr>
inline return_type_t<T_prob> categorical_lpmf(const std::vector<int>& ns,
                                              const T_prob& theta) {
  using T_partials_return = partials_return_t<T_prob>;
  static constexpr const char* function = "categorical_lpmf";

  check_bounded(function, "element of outcome array", ns, 1, theta.size());
  ref_type_t<T_prob> theta_ref = theta;
  const auto& theta_val = to_ref(value_of(theta_ref));
  check_simplex(function, "Probabilities parameter", theta_val);

  if constexpr (!include_summand<propto, T_prob>::value) {
    return 0.0;
  }

  if (ns.size() == 0) {
    return 0.0;
  }

  // Count the outcomes per category and take the log of the observed
  // entries only. The log of every entry would leave a log(0) node for an
  // unobserved zero entry, whose reverse pass gives 0 / 0 = NaN.
  Eigen::VectorXi counts = Eigen::VectorXi::Zero(theta.size());
  for (int n : ns) {
    ++counts.coeffRef(n - 1);
  }

  auto ops_partials = make_partials_propagator(theta_ref);
  T_partials_return lp(0.0);
  for (Eigen::Index k = 0; k < counts.size(); ++k) {
    if (counts.coeff(k) > 0) {
      lp += counts.coeff(k) * log(theta_val.coeff(k));
      if constexpr (is_autodiff_v<T_prob>) {
        partials<0>(ops_partials)[k] = counts.coeff(k) / theta_val.coeff(k);
      }
    }
  }
  return ops_partials.build(lp);
}

template <typename T_n, typename T_prob, require_st_integral<T_n>* = nullptr,
          require_eigen_col_vector_t<T_prob>* = nullptr>
inline return_type_t<T_prob> categorical_lpmf(const T_n& ns,
                                              const T_prob& theta) {
  return categorical_lpmf<false>(ns, theta);
}

}  // namespace math
}  // namespace stan
#endif
