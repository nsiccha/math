#ifndef STAN_MATH_REV_FUNCTOR_ODE_USER_JACOBIAN_HPP
#define STAN_MATH_REV_FUNCTOR_ODE_USER_JACOBIAN_HPP

#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/fun/value_of.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/prim/functor/apply.hpp>
#include <stan/math/prim/functor/ode_jacobian_pair.hpp>
#include <algorithm>
#include <cmath>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <vector>

#ifndef STAN_MATH_USER_JACOBIAN_TOLERANCE
/**
 * Largest accepted difference, relative to max(1, |autodiff entry|),
 * between a user Jacobian and the autodiff Jacobian when
 * <code>STAN_MATH_CHECK_USER_JACOBIAN</code> is defined.
 */
#define STAN_MATH_USER_JACOBIAN_TOLERANCE 1e-6
#endif

namespace stan {
namespace math {
namespace internal {

/**
 * Copy of x with every real scalar promoted to an autodiff variable;
 * integers stay integers. Reals are visited in the order of
 * <code>ode_jacobian_var_columns</code>.
 */
template <typename T>
inline auto promote_reals_to_var(const T& x) {
  using T_plain = std::decay_t<T>;
  if constexpr (is_tuple<T_plain>::value) {
    return math::apply(
        [](const auto&... inner) {
          return std::make_tuple(promote_reals_to_var(inner)...);
        },
        x);
  } else if constexpr (std::is_integral<scalar_type_t<T_plain>>::value) {
    return T_plain(x);
  } else if constexpr (is_std_vector<T_plain>::value) {
    using elem_t = decltype(promote_reals_to_var(
        std::declval<const value_type_t<T_plain>&>()));
    std::vector<elem_t> out;
    out.reserve(x.size());
    for (const auto& x_i : x) {
      out.push_back(promote_reals_to_var(x_i));
    }
    return out;
  } else if constexpr (is_eigen<T_plain>::value) {
    return plain_type_t<decltype(value_of(x).template cast<var>())>(
        value_of(x).template cast<var>());
  } else {
    return var(value_of(x));
  }
}

/**
 * Write the adjoints of every autodiff scalar of x to dest, in the order of
 * <code>ode_jacobian_var_columns</code>, and advance dest.
 */
template <typename T>
inline void collect_var_adjoints(const T& x, double*& dest) {
  using T_plain = std::decay_t<T>;
  if constexpr (is_tuple<T_plain>::value) {
    math::apply(
        [&](const auto&... inner) { (collect_var_adjoints(inner, dest), ...); },
        x);
  } else if constexpr (is_std_vector<T_plain>::value) {
    for (const auto& x_i : x) {
      collect_var_adjoints(x_i, dest);
    }
  } else if constexpr (is_eigen<T_plain>::value) {
    if constexpr (is_var<value_type_t<T_plain>>::value) {
      for (Eigen::Index i = 0; i < x.size(); ++i) {
        *dest++ = x.coeff(i).adj();
      }
    }
  } else if constexpr (is_var<T_plain>::value) {
    *dest++ = x.adj();
  }
}

/**
 * Copy of an argument with its real scalars promoted to autodiff variables
 * if PromoteAll is set or the argument's original type Orig has autodiff
 * scalars; otherwise an unchanged copy.
 */
template <typename Orig, bool PromoteAll, typename T>
inline auto promote_arg_to_var(const T& x) {
  if constexpr (PromoteAll || is_var<scalar_type_t<Orig>>::value) {
    return promote_reals_to_var(x);
  } else {
    return std::decay_t<T>(x);
  }
}

template <bool PromoteAll, typename... Origs, typename... Args, size_t... I>
inline auto promote_args_to_var(const std::tuple<Args...>& args,
                                std::index_sequence<I...>) {
  return std::make_tuple(
      promote_arg_to_var<Origs, PromoteAll>(std::get<I>(args))...);
}

/**
 * The Jacobian of f(t, states..., msgs, args...) by reverse-mode autodiff
 * in the user-Jacobian layout: one row per component of f, one column per
 * scalar of the states (in order, Eigen types column-major), then one per
 * real scalar of the promoted arguments (see promote_arg_to_var).
 *
 * @tparam PromoteAll whether to differentiate with respect to every real
 *   argument, or only those whose original type has autodiff scalars
 * @tparam Origs original types of the variadic arguments
 * @param f right-hand side or residual functor
 * @param t time
 * @param states the states f differentiates with respect to (y; or yy, yp;
 *   or y, y_lag), as doubles
 * @param msgs message stream
 * @param args the variadic arguments, as doubles
 */
template <bool PromoteAll, typename... Origs, typename F, typename... States,
          typename... Args>
inline Eigen::MatrixXd autodiff_user_jacobian(
    const F& f, double t, const std::tuple<States...>& states,
    std::ostream* msgs, const std::tuple<Args...>& args) {
  static_assert(sizeof...(Origs) == sizeof...(Args),
                "one original type per argument");
  nested_rev_autodiff nested;
  auto states_v = math::apply(
      [](const auto&... s) {
        return std::make_tuple(promote_reals_to_var(s)...);
      },
      states);
  auto args_v = promote_args_to_var<PromoteAll, Origs...>(
      args, std::index_sequence_for<Args...>{});
  Eigen::Matrix<var, Eigen::Dynamic, 1> fy = math::apply(
      [&](const auto&... s) {
        return math::apply(
            [&](const auto&... a) { return f(t, s..., msgs, a...); }, args_v);
      },
      states_v);
  const size_t num_state_columns = math::apply(
      [](const auto&... s) { return (static_cast<size_t>(s.size()) + ...); },
      states_v);
  size_t num_arg_columns = 0;
  const size_t num_arg_vars = math::apply(
      [&](const auto&... a) {
        return ode_jacobian_var_columns(num_arg_columns, a...).size();
      },
      args_v);
  const size_t num_columns = num_state_columns + num_arg_vars;
  Eigen::MatrixXd jac(fy.size(), num_columns);
  Eigen::VectorXd row(num_columns);
  for (Eigen::Index i = 0; i < fy.size(); ++i) {
    if (i > 0) {
      nested.set_zero_all_adjoints();
    }
    fy.coeffRef(i).grad();
    double* dest = row.data();
    math::apply([&](const auto&... s) { (collect_var_adjoints(s, dest), ...); },
                states_v);
    math::apply([&](const auto&... a) { (collect_var_adjoints(a, dest), ...); },
                args_v);
    jac.row(i) = row.transpose();
  }
  return jac;
}

/**
 * The state columns of a user Jacobian followed by the given argument
 * columns.
 */
inline Eigen::MatrixXd select_user_jacobian_columns(
    const Eigen::MatrixXd& jac, size_t num_state_columns,
    const std::vector<size_t>& arg_columns) {
  Eigen::MatrixXd selected(jac.rows(), num_state_columns + arg_columns.size());
  selected.leftCols(num_state_columns) = jac.leftCols(num_state_columns);
  for (size_t k = 0; k < arg_columns.size(); ++k) {
    selected.col(num_state_columns + k)
        = jac.col(num_state_columns + arg_columns[k]);
  }
  return selected;
}

/**
 * Largest difference between two Jacobians, relative to max(1, |entry of
 * the reference|).
 */
inline double user_jacobian_difference(const Eigen::MatrixXd& user,
                                       const Eigen::MatrixXd& reference) {
  if (user.size() == 0) {
    return 0;
  }
  return ((user - reference).array().abs() / reference.array().abs().max(1.0))
      .maxCoeff();
}

template <typename... States>
inline size_t user_jacobian_state_columns(const std::tuple<States...>& states) {
  return math::apply(
      [](const auto&... s) { return (static_cast<size_t>(s.size()) + ...); },
      states);
}

/**
 * Evaluate the user Jacobian of a right-hand side paired with it (see
 * <code>ode_jacobian_pair</code>) and check its size. With
 * <code>STAN_MATH_CHECK_USER_JACOBIAN</code> defined, also compare the
 * columns the solvers read (states and autodiff arguments) with autodiff
 * and throw if they differ by more than
 * <code>STAN_MATH_USER_JACOBIAN_TOLERANCE</code>.
 *
 * @tparam Origs original types of the variadic arguments
 * @param function name of the calling function, for error messages
 * @param f pair of right-hand side and Jacobian functors
 * @param t time
 * @param states the states (y; or yy, yp; or y, y_lag), as doubles
 * @param msgs message stream
 * @param args the variadic arguments, as doubles
 * @param num_arg_columns number of real scalars of the arguments
 * @param var_columns argument columns of the autodiff scalars
 * @return the user Jacobian
 * @throw std::invalid_argument if the user Jacobian has the wrong size
 * @throw std::domain_error if checking and it differs from autodiff
 */
template <typename... Origs, typename F, typename FJac, typename... States,
          typename... Args>
inline Eigen::MatrixXd eval_user_jacobian(
    const std::tuple<Origs...>* /* original argument types */,
    const char* function, const ode_jacobian_pair<F, FJac>& f, double t,
    const std::tuple<States...>& states, std::ostream* msgs,
    const std::tuple<Args...>& args, size_t num_arg_columns,
    const std::vector<size_t>& var_columns) {
  Eigen::MatrixXd jac = math::apply(
      [&](const auto&... s) {
        return math::apply(
            [&](const auto&... a) { return f.jacobian(t, s..., msgs, a...); },
            args);
      },
      states);
  const size_t num_states = std::get<0>(states).rows();
  const size_t num_state_columns = user_jacobian_state_columns(states);
  check_size_match(function, "rows of the Jacobian", jac.rows(), "states",
                   num_states);
  check_size_match(function, "columns of the Jacobian", jac.cols(),
                   "state and argument scalars",
                   num_state_columns + num_arg_columns);
#ifdef STAN_MATH_CHECK_USER_JACOBIAN
  const double difference = user_jacobian_difference(
      select_user_jacobian_columns(jac, num_state_columns, var_columns),
      autodiff_user_jacobian<false, Origs...>(f.f_, t, states, msgs, args));
  if (difference > STAN_MATH_USER_JACOBIAN_TOLERANCE) {
    throw_domain_error(function, "the user Jacobian", difference,
                       "differs from the autodiff Jacobian by ",
                       " (largest difference relative to max(1, |autodiff "
                       "entry|))");
  }
#endif
  return jac;
}

/**
 * Evaluate the user pullback lambda^T [df/dy | df/dargs] of a right-hand
 * side paired with it (see <code>ode_jacobian_pullback_pair</code>) and
 * check its size. With <code>STAN_MATH_CHECK_USER_JACOBIAN</code> defined,
 * also compare the entries the solver reads (states and autodiff
 * arguments) with the autodiff Jacobian applied to lambda.
 *
 * @return the vector of length N + P
 */
template <typename... Origs, typename F, typename FJac, typename FPull,
          typename... Args>
inline Eigen::VectorXd eval_user_pullback(
    const std::tuple<Origs...>* /* original argument types */,
    const char* function, const ode_jacobian_pullback_pair<F, FJac, FPull>& f,
    double t, const Eigen::VectorXd& y, const Eigen::VectorXd& lambda,
    std::ostream* msgs, const std::tuple<Args...>& args, size_t num_arg_columns,
    const std::vector<size_t>& var_columns) {
  Eigen::VectorXd g = math::apply(
      [&](const auto&... a) { return f.pullback(t, y, lambda, msgs, a...); },
      args);
  check_size_match(function, "size of the pullback", g.size(),
                   "states plus argument scalars", y.size() + num_arg_columns);
#ifdef STAN_MATH_CHECK_USER_JACOBIAN
  const Eigen::MatrixXd jac_ad = autodiff_user_jacobian<false, Origs...>(
      f.f_, t, std::tie(y), msgs, args);
  const Eigen::MatrixXd g_sel
      = select_user_jacobian_columns(g.transpose(), y.size(), var_columns);
  const double difference
      = user_jacobian_difference(g_sel, (lambda.transpose() * jac_ad).eval());
  if (difference > STAN_MATH_USER_JACOBIAN_TOLERANCE) {
    throw_domain_error(function, "the user pullback", difference,
                       "differs from the autodiff pullback by ",
                       " (largest difference relative to max(1, |autodiff "
                       "entry|))");
  }
#endif
  return g;
}

/**
 * Whether f(t, states..., msgs, args...) is callable with the element types
 * of the two tuples.
 */
template <typename F, typename StatesTuple, typename ArgsTuple>
struct invocable_with_tuples;

template <typename F, typename... S, typename... A>
struct invocable_with_tuples<F, std::tuple<S...>, std::tuple<A...>>
    : std::is_invocable<const F&, double, const S&..., std::ostream*,
                        const A&...> {};

/**
 * The tuple type Tuple with its I-th element's real scalars promoted to
 * autodiff variables.
 */
template <size_t I, typename Tuple, typename Seq>
struct promote_tuple_element;

template <size_t I, typename... A, size_t... J>
struct promote_tuple_element<I, std::tuple<A...>, std::index_sequence<J...>> {
  using type = std::tuple<std::conditional_t<
      I == J, decltype(promote_reals_to_var(std::declval<const A&>())), A>...>;
};

/**
 * Original-type markers for checking a user Jacobian at one point: an
 * argument is differentiated (marker var) if it was passed as an autodiff
 * variable or f accepts autodiff variables in its place; otherwise it stays
 * data (marker double).
 */
template <typename F, typename StatesV, typename ArgsD, typename Origs,
          typename Seq>
struct user_jacobian_check_markers;

template <typename F, typename StatesV, typename... A, typename... Origs,
          size_t... I>
struct user_jacobian_check_markers<F, StatesV, std::tuple<A...>,
                                   std::tuple<Origs...>,
                                   std::index_sequence<I...>> {
  using type = std::tuple<std::conditional_t<
      is_var<scalar_type_t<Origs>>::value
          || invocable_with_tuples<
              F, StatesV,
              typename promote_tuple_element<
                  I, std::tuple<A...>, std::index_sequence<I...>>::type>::value,
      var, double>...>;
};

/**
 * Largest difference between a user Jacobian and the autodiff Jacobian at
 * one point, relative to max(1, |autodiff entry|). Compares the state
 * columns and the columns of every argument passed as an autodiff variable
 * or accepted as one by f; real data arguments that f takes only as data
 * cannot be differentiated and their columns are not compared (the
 * solvers never read them).
 */
template <typename F, typename FJac, typename... States, typename... Args>
inline double user_jacobian_error(const char* function, const F& f,
                                  const FJac& jac, double t,
                                  const std::tuple<States...>& states,
                                  std::ostream* msgs, const Args&... args) {
  const auto args_d = std::make_tuple(eval(value_of(args))...);
  size_t num_arg_columns = 0;
  ode_jacobian_var_columns(num_arg_columns, args...);
  const ode_jacobian_pair<F, FJac> f_jac(f, jac);
  const Eigen::MatrixXd user = math::apply(
      [&](const auto&... s) {
        return math::apply(
            [&](const auto&... a) {
              return f_jac.jacobian(t, s..., msgs, a...);
            },
            args_d);
      },
      states);
  const size_t num_states = std::get<0>(states).rows();
  const size_t num_state_columns = user_jacobian_state_columns(states);
  check_size_match(function, "rows of the Jacobian", user.rows(), "states",
                   num_states);
  check_size_match(function, "columns of the Jacobian", user.cols(),
                   "state and argument scalars",
                   num_state_columns + num_arg_columns);
  using states_v_t = std::tuple<decltype(promote_reals_to_var(
      std::declval<const std::decay_t<States>&>()))...>;
  using args_d_t = std::decay_t<decltype(args_d)>;
  using markers_t = typename user_jacobian_check_markers<
      F, states_v_t, args_d_t, std::tuple<Args...>,
      std::index_sequence_for<Args...>>::type;
  return math::apply(
      [&](auto... marker) {
        // the columns of the differentiated arguments
        std::vector<size_t> compared;
        {
          nested_rev_autodiff nested;
          const auto args_v = promote_args_to_var<false, decltype(marker)...>(
              args_d, std::index_sequence_for<Args...>{});
          size_t n = 0;
          compared = math::apply(
              [&](const auto&... a) {
                return ode_jacobian_var_columns(n, a...);
              },
              args_v);
        }
        return user_jacobian_difference(
            select_user_jacobian_columns(user, num_state_columns, compared),
            autodiff_user_jacobian<false, decltype(marker)...>(f, t, states,
                                                               msgs, args_d));
      },
      markers_t{});
}

}  // namespace internal

/**
 * Check a user-supplied ODE Jacobian against autodiff at one point.
 *
 * @param f right-hand side of the ODE, as for <code>ode_rk45</code>
 * @param jac its Jacobian, as for the <code>ode_rk45</code> overload
 *   taking a Jacobian
 * @param t time
 * @param y state
 * @param msgs message stream
 * @param args variadic arguments of f
 * @return the largest difference between jac and the autodiff Jacobian of
 *   f at (t, y), relative to max(1, |autodiff entry|)
 * @throw std::invalid_argument if jac returns a matrix of the wrong size
 */
template <typename F, typename FJac, typename T_t, typename T_y,
          typename... Args, require_eigen_col_vector_t<T_y>* = nullptr>
inline double ode_jacobian_error(const F& f, const FJac& jac, const T_t& t,
                                 const T_y& y, std::ostream* msgs,
                                 const Args&... args) {
  const Eigen::VectorXd y_d = value_of(y);
  return internal::user_jacobian_error("ode_jacobian_error", f, jac,
                                       value_of(t), std::make_tuple(y_d), msgs,
                                       args...);
}

/**
 * Check a user-supplied DAE residual Jacobian against autodiff at one
 * point; see <code>ode_jacobian_error</code>.
 *
 * @param f residual of the DAE, as for <code>dae</code>
 * @param jac its Jacobian, as for the <code>dae</code> overload taking a
 *   Jacobian
 * @param t time
 * @param yy state
 * @param yp time derivative of the state
 * @param msgs message stream
 * @param args variadic arguments of f
 * @return the largest relative difference
 */
template <typename F, typename FJac, typename T_t, typename T_yy, typename T_yp,
          typename... Args,
          require_all_eigen_col_vector_t<T_yy, T_yp>* = nullptr>
inline double dae_jacobian_error(const F& f, const FJac& jac, const T_t& t,
                                 const T_yy& yy, const T_yp& yp,
                                 std::ostream* msgs, const Args&... args) {
  const Eigen::VectorXd yy_d = value_of(yy);
  const Eigen::VectorXd yp_d = value_of(yp);
  return internal::user_jacobian_error("dae_jacobian_error", f, jac,
                                       value_of(t), std::make_tuple(yy_d, yp_d),
                                       msgs, args...);
}

/**
 * Check a user-supplied DDE Jacobian against autodiff at one point; see
 * <code>ode_jacobian_error</code>.
 *
 * @param f right-hand side of the DDE, as for <code>dde_rk45</code>
 * @param jac its Jacobian, as for the <code>dde_rk45</code> overload
 *   taking a Jacobian
 * @param t time
 * @param y state
 * @param y_lag lagged states, one column per delay
 * @param msgs message stream
 * @param args variadic arguments of f
 * @return the largest relative difference
 */
template <typename F, typename FJac, typename T_t, typename T_y, typename T_lag,
          typename... Args, require_eigen_col_vector_t<T_y>* = nullptr,
          require_eigen_t<T_lag>* = nullptr>
inline double dde_jacobian_error(const F& f, const FJac& jac, const T_t& t,
                                 const T_y& y, const T_lag& y_lag,
                                 std::ostream* msgs, const Args&... args) {
  const Eigen::VectorXd y_d = value_of(y);
  const Eigen::MatrixXd y_lag_d = value_of(y_lag);
  return internal::user_jacobian_error(
      "dde_jacobian_error", f, jac, value_of(t), std::make_tuple(y_d, y_lag_d),
      msgs, args...);
}

}  // namespace math
}  // namespace stan

#endif
