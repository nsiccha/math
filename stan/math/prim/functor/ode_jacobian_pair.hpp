#ifndef STAN_MATH_PRIM_FUNCTOR_ODE_JACOBIAN_PAIR_HPP
#define STAN_MATH_PRIM_FUNCTOR_ODE_JACOBIAN_PAIR_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/functor/apply.hpp>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace stan {
namespace math {
namespace internal {

/**
 * Pairs the right-hand side functor of a differential equation with a
 * user-supplied functor for its Jacobian.
 *
 * <p>Calling the pair calls the right-hand side, so the pair can stand in
 * for the right-hand side functor wherever a solver evaluates it. The
 * coupled sensitivity systems detect the pair (see
 * <code>is_ode_jacobian_pair</code>) and evaluate the user Jacobian in
 * doubles instead of computing it by nested reverse-mode autodiff.
 *
 * <p>The Jacobian functor takes the same arguments as the right-hand side
 * and returns a matrix with one row per state and one column per state
 * followed by one column per real scalar of the variadic arguments, in the
 * order <code>count_vars</code> visits them (arguments left to right,
 * Eigen types column-major, arrays element by element, tuples element by
 * element). Columns belonging to real data arguments are not read; integer
 * arguments get no columns.
 *
 * @tparam F type of the right-hand side functor
 * @tparam FJac type of the Jacobian functor
 */
template <typename F, typename FJac>
struct ode_jacobian_pair {
  // held by value: the adjoint solver keeps its functor beyond the call
  const std::decay_t<F> f_;
  const std::decay_t<FJac> jac_;

  ode_jacobian_pair(const F& f, const FJac& jac) : f_(f), jac_(jac) {}

  /**
   * Evaluate the right-hand side.
   */
  template <typename... Args>
  inline auto operator()(Args&&... args) const {
    return f_(std::forward<Args>(args)...);
  }

  /**
   * Evaluate the user-supplied Jacobian.
   */
  template <typename... Args>
  inline auto jacobian(Args&&... args) const {
    return jac_(std::forward<Args>(args)...);
  }
};

template <typename T>
struct is_ode_jacobian_pair : std::false_type {};

template <typename F, typename FJac>
struct is_ode_jacobian_pair<ode_jacobian_pair<F, FJac>> : std::true_type {};

/**
 * Walk one argument in the order <code>count_vars</code> visits its
 * scalars, recording for every autodiff scalar the index of its column in
 * the parameter block of a user Jacobian.
 *
 * @param x argument to walk
 * @param[in, out] column index of the next scalar's column
 * @param[in, out] var_columns column indices of the autodiff scalars seen
 */
template <typename T>
inline void ode_jacobian_var_columns_impl(
    const T& x, std::size_t& column, std::vector<std::size_t>& var_columns) {
  using T_plain = std::decay_t<T>;
  if constexpr (is_tuple<T_plain>::value) {
    math::apply(
        [&](const auto&... inner) {
          (ode_jacobian_var_columns_impl(inner, column, var_columns), ...);
        },
        x);
  } else if constexpr (is_std_vector<T_plain>::value) {
    for (const auto& x_i : x) {
      ode_jacobian_var_columns_impl(x_i, column, var_columns);
    }
  } else if constexpr (is_eigen<T_plain>::value) {
    const std::size_t size = x.size();
    if constexpr (is_var<value_type_t<T_plain>>::value) {
      for (std::size_t i = 0; i < size; ++i) {
        var_columns.push_back(column + i);
      }
    }
    column += size;
  } else if constexpr (is_var<T_plain>::value) {
    var_columns.push_back(column);
    ++column;
  } else if constexpr (std::is_integral<T_plain>::value) {
    // integer arguments are always data and get no columns
  } else {
    static_assert(std::is_floating_point<T_plain>::value,
                  "ODE arguments must be scalars, Eigen types, std::vectors "
                  "or tuples of them");
    ++column;
  }
}

/**
 * Columns of the parameter block of a user Jacobian that hold the
 * derivatives with respect to the autodiff scalars of the arguments.
 *
 * @param[out] num_columns number of real scalars of all arguments, i.e.
 *   the number of columns of the parameter block
 * @param args variadic arguments of the right-hand side
 * @return for the k-th autodiff scalar (in <code>count_vars</code> order)
 *   the index of its column within the parameter block
 */
template <typename... Args>
inline std::vector<std::size_t> ode_jacobian_var_columns(
    std::size_t& num_columns, const Args&... args) {
  std::vector<std::size_t> var_columns;
  num_columns = 0;
  (ode_jacobian_var_columns_impl(args, num_columns, var_columns), ...);
  return var_columns;
}

}  // namespace internal
}  // namespace math
}  // namespace stan

#endif
