// Benchmark: push-forward (J D) and pullback (J^T lambda) user functions
// against the dense user Jacobian and nested autodiff.
//
// Forward sensitivities need J_y S + J_theta, a push-forward of the
// sensitivity directions; the adjoint method needs lambda^T J, a pullback.
// The push-forward is wired into ode_rk45 through a benchmark-local
// coupled_ode_system specialization (the hook a library version would use)
// and timed end to end; the per-evaluation costs of the forward sensitivity
// right-hand side and of the adjoint backward step (J_y^T lambda and
// J_theta^T lambda) are timed directly.
#include "models.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using stan::math::var;

/** Pairs a right-hand side with a push-forward functor (t, y, D, theta). */
template <typename F, typename FPush>
struct push_pair {
  F f_;
  FPush push_;
  template <typename... A>
  auto operator()(A&&... a) const {
    return f_(std::forward<A>(a)...);
  }
};

/** Pairs a right-hand side with a transposed push-forward functor. */
template <typename F, typename FPush>
struct push_t_pair {
  F f_;
  FPush push_;
  template <typename... A>
  auto operator()(A&&... a) const {
    return f_(std::forward<A>(a)...);
  }
};

namespace stan {
namespace math {
/**
 * Coupled forward-sensitivity system whose sensitivity right-hand side is
 * one push-forward of D = [S; W], with S the state sensitivities and W the
 * fixed selection of the autodiff argument scalars.
 */
template <typename F, typename FPush, typename T_y0, typename... Args>
struct coupled_ode_system_impl<false, push_pair<F, FPush>, T_y0, Args...> {
  const push_pair<F, FPush>& f_;
  const Eigen::Matrix<T_y0, -1, 1>& y0_;
  std::tuple<plain_type_t<decltype(value_of(std::declval<const Args&>()))>...>
      args_;
  const size_t num_y0_vars_;
  const size_t num_args_vars_;
  const size_t N_;
  mutable MatrixXd D_;

  coupled_ode_system_impl(const push_pair<F, FPush>& f,
                          const Eigen::Matrix<T_y0, -1, 1>& y0,
                          std::ostream* msgs, const Args&... args)
      : f_(f),
        y0_(y0),
        args_(value_of(args)...),
        num_y0_vars_(count_vars(y0)),
        num_args_vars_(count_vars(args...)),
        N_(y0.size()) {
    size_t num_arg_columns = 0;
    const std::vector<size_t> var_columns
        = internal::ode_jacobian_var_columns(num_arg_columns, args...);
    const size_t K = num_y0_vars_ + num_args_vars_;
    D_ = MatrixXd::Zero(N_ + num_arg_columns, K);
    for (size_t k = 0; k < num_args_vars_; ++k) {
      D_(N_ + var_columns[k], num_y0_vars_ + k) = 1;
    }
  }

  void operator()(const std::vector<double>& z, std::vector<double>& dz_dt,
                  double t) const {
    dz_dt.resize(size());
    const VectorXd y = Eigen::Map<const VectorXd>(z.data(), N_);
    Eigen::Map<VectorXd>(dz_dt.data(), N_) = math::apply(
        [&](auto&&... a) { return f_.f_(t, y, nullptr, a...); }, args_);
    const size_t K = num_y0_vars_ + num_args_vars_;
    D_.topRows(N_) = Eigen::Map<const MatrixXd>(z.data() + N_, N_, K);
    Eigen::Map<MatrixXd>(dz_dt.data() + N_, N_, K) = math::apply(
        [&](auto&&... a) { return f_.push_(t, y, D_, nullptr, a...); }, args_);
  }

  size_t size() const { return N_ + N_ * (num_y0_vars_ + num_args_vars_); }

  std::vector<double> initial_state() const {
    std::vector<double> initial(size(), 0.0);
    for (size_t i = 0; i < N_; i++) {
      initial[i] = value_of(y0_(i));
    }
    for (size_t i = 0; i < num_y0_vars_; i++) {
      initial[N_ + i * N_ + i] = 1.0;
    }
    return initial;
  }
};
/**
 * The same with a transposed push-forward (t, y, Dt, out, theta): Dt is
 * K x (N + P) and out K x N, so the generated terms are contiguous.
 */
template <typename F, typename FPush, typename T_y0, typename... Args>
struct coupled_ode_system_impl<false, push_t_pair<F, FPush>, T_y0, Args...> {
  const push_t_pair<F, FPush>& f_;
  const Eigen::Matrix<T_y0, -1, 1>& y0_;
  std::tuple<plain_type_t<decltype(value_of(std::declval<const Args&>()))>...>
      args_;
  const size_t num_y0_vars_;
  const size_t num_args_vars_;
  const size_t N_;
  mutable MatrixXd Dt_;
  mutable MatrixXd out_;

  coupled_ode_system_impl(const push_t_pair<F, FPush>& f,
                          const Eigen::Matrix<T_y0, -1, 1>& y0,
                          std::ostream* msgs, const Args&... args)
      : f_(f),
        y0_(y0),
        args_(value_of(args)...),
        num_y0_vars_(count_vars(y0)),
        num_args_vars_(count_vars(args...)),
        N_(y0.size()) {
    size_t num_arg_columns = 0;
    const std::vector<size_t> var_columns
        = internal::ode_jacobian_var_columns(num_arg_columns, args...);
    const size_t K = num_y0_vars_ + num_args_vars_;
    Dt_ = MatrixXd::Zero(K, N_ + num_arg_columns);
    out_ = MatrixXd(K, N_);
    for (size_t k = 0; k < num_args_vars_; ++k) {
      Dt_(num_y0_vars_ + k, N_ + var_columns[k]) = 1;
    }
  }

  void operator()(const std::vector<double>& z, std::vector<double>& dz_dt,
                  double t) const {
    dz_dt.resize(size());
    const VectorXd y = Eigen::Map<const VectorXd>(z.data(), N_);
    Eigen::Map<VectorXd>(dz_dt.data(), N_) = math::apply(
        [&](auto&&... a) { return f_.f_(t, y, nullptr, a...); }, args_);
    const size_t K = num_y0_vars_ + num_args_vars_;
    Dt_.leftCols(N_)
        = Eigen::Map<const MatrixXd>(z.data() + N_, N_, K).transpose();
    math::apply([&](auto&&... a) { f_.push_(t, y, Dt_, out_, nullptr, a...); },
                args_);
    Eigen::Map<MatrixXd>(dz_dt.data() + N_, N_, K) = out_.transpose();
  }

  size_t size() const { return N_ + N_ * (num_y0_vars_ + num_args_vars_); }

  std::vector<double> initial_state() const {
    std::vector<double> initial(size(), 0.0);
    for (size_t i = 0; i < N_; i++) {
      initial[i] = value_of(y0_(i));
    }
    for (size_t i = 0; i < num_y0_vars_; i++) {
      initial[N_ + i * N_ + i] = 1.0;
    }
    return initial;
  }
};
}  // namespace math
}  // namespace stan

struct Result {
  std::vector<double> vals;
  std::vector<double> grads;
};

enum class Path { ad, jac, push, pusht };

template <Path P, typename Rhs, typename Jac, typename Push, typename PushT>
Result run_once(const VectorXd& y0, bool y0_var, const std::vector<double>& ts,
                double tol, const VectorXd& theta) {
  Result res;
  {
    Eigen::Matrix<var, -1, 1> theta_v = theta.template cast<var>();
    Eigen::Matrix<var, -1, 1> y0_v = y0.template cast<var>();
    auto solve = [&](const auto& y0_a) {
      if constexpr (P == Path::ad) {
        return stan::math::ode_rk45_tol(Rhs(), y0_a, 0.0, ts, tol, tol,
                                        100000000, nullptr, theta_v);
      } else if constexpr (P == Path::jac) {
        return stan::math::ode_rk45_tol(Rhs(), Jac(), y0_a, 0.0, ts, tol, tol,
                                        100000000, nullptr, theta_v);
      } else if constexpr (P == Path::pusht) {
        const push_t_pair<Rhs, PushT> f_push{Rhs(), PushT()};
        return stan::math::ode_rk45_tol_impl("pusht", f_push, y0_a, 0.0, ts,
                                             tol, tol, 100000000, nullptr,
                                             theta_v);
      } else {
        const push_pair<Rhs, Push> f_push{Rhs(), Push()};
        return stan::math::ode_rk45_tol_impl("push", f_push, y0_a, 0.0, ts, tol,
                                             tol, 100000000, nullptr, theta_v);
      }
    };
    std::vector<Eigen::Matrix<var, -1, 1>> sol
        = y0_var ? solve(y0_v) : solve(y0);
    var lp = 0;
    for (size_t i = 0; i < sol.size(); ++i) {
      for (int k = 0; k < sol[i].size(); ++k) {
        lp += (1.0 + 0.1 * std::sin(1.0 + i + 7.0 * k)) * sol[i](k);
        res.vals.push_back(sol[i](k).val());
      }
    }
    lp.grad();
    for (int j = 0; j < theta_v.size(); ++j) {
      res.grads.push_back(theta_v(j).adj());
    }
    if (y0_var) {
      for (int j = 0; j < y0_v.size(); ++j) {
        res.grads.push_back(y0_v(j).adj());
      }
    }
  }
  stan::math::recover_memory();
  return res;
}

double median(std::vector<double> v) {
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}

double max_rel_diff(const std::vector<double>& a, const std::vector<double>& b,
                    double floor) {
  double m = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::abs(a[i] - b[i])
                        / std::max({std::abs(a[i]), std::abs(b[i]), floor}));
  }
  return m;
}

template <typename Fn>
double seconds(const Fn& fn) {
  auto start = std::chrono::steady_clock::now();
  fn();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

/** End-to-end rk45 gradient plus the per-evaluation forward RHS costs. */
template <typename Rhs, typename Jac, typename Push, typename PushT>
void bench_forward(const std::string& name, const VectorXd& y0, bool y0_var,
                   const std::vector<double>& ts, const VectorXd& theta,
                   double budget_s) {
  const double tol = 1e-6;
  auto ad = [&] {
    return run_once<Path::ad, Rhs, Jac, Push, PushT>(y0, y0_var, ts, tol,
                                                     theta);
  };
  auto jac = [&] {
    return run_once<Path::jac, Rhs, Jac, Push, PushT>(y0, y0_var, ts, tol,
                                                      theta);
  };
  auto push = [&] {
    return run_once<Path::push, Rhs, Jac, Push, PushT>(y0, y0_var, ts, tol,
                                                       theta);
  };
  auto pusht = [&] {
    return run_once<Path::pusht, Rhs, Jac, Push, PushT>(y0, y0_var, ts, tol,
                                                        theta);
  };
  const Result r_ad = ad(), r_jac = jac(), r_push = push(), r_pusht = pusht();
  const double first
      = seconds(ad) + seconds(jac) + seconds(push) + seconds(pusht);
  const int reps
      = std::max(7, std::min(2000, static_cast<int>(budget_s / first)));
  std::vector<double> t_ad, t_jac, t_push, t_pusht;
  for (int r = 0; r < reps; ++r) {
    t_ad.push_back(seconds(ad));
    t_jac.push_back(seconds(jac));
    t_push.push_back(seconds(push));
    t_pusht.push_back(seconds(pusht));
  }
  // per-evaluation coupled right-hand side at the initial state
  Rhs f;
  Jac j;
  const stan::math::internal::ode_jacobian_pair<Rhs, Jac> f_jac(f, j);
  const push_pair<Rhs, Push> f_push{Rhs(), Push()};
  const push_t_pair<Rhs, PushT> f_pusht{Rhs(), PushT()};
  Eigen::Matrix<var, -1, 1> theta_v = theta.template cast<var>();
  std::vector<double> e_ad, e_jac, e_push, e_pusht;
  auto per_eval = [&](const auto& y0_a) {
    using T_y0 = typename std::decay_t<decltype(y0_a)>::Scalar;
    using A = Eigen::Matrix<var, -1, 1>;
    stan::math::coupled_ode_system<Rhs, T_y0, A> c_ad(f, y0_a, nullptr,
                                                      theta_v);
    stan::math::coupled_ode_system<std::decay_t<decltype(f_jac)>, T_y0, A>
        c_jac(f_jac, y0_a, nullptr, theta_v);
    stan::math::coupled_ode_system<std::decay_t<decltype(f_push)>, T_y0, A>
        c_push(f_push, y0_a, nullptr, theta_v);
    stan::math::coupled_ode_system<std::decay_t<decltype(f_pusht)>, T_y0, A>
        c_pusht(f_pusht, y0_a, nullptr, theta_v);
    std::vector<double> z = c_ad.initial_state();
    for (size_t i = y0.size(); i < z.size(); ++i) {
      z[i] += 0.01 * std::sin(static_cast<double>(i));
    }
    std::vector<double> d1, d2, d3, d4;
    for (int r = 0; r < 2001; ++r) {
      e_ad.push_back(seconds([&] { c_ad(z, d1, 0.5); }));
      e_jac.push_back(seconds([&] { c_jac(z, d2, 0.5); }));
      e_push.push_back(seconds([&] { c_push(z, d3, 0.5); }));
      e_pusht.push_back(seconds([&] { c_pusht(z, d4, 0.5); }));
    }
    double diff = 0;
    for (size_t i = 0; i < z.size(); ++i) {
      diff = std::max(diff,
                      std::abs(d1[i] - d3[i]) / std::max(1.0, std::abs(d1[i])));
      diff = std::max(diff,
                      std::abs(d1[i] - d4[i]) / std::max(1.0, std::abs(d1[i])));
    }
    if (diff > 1e-10) {
      std::cout << "push-forward RHS differs from AD: " << diff << std::endl;
    }
  };
  if (y0_var) {
    Eigen::Matrix<var, -1, 1> y0_v = y0.template cast<var>();
    per_eval(y0_v);
  } else {
    per_eval(y0);
  }
  stan::math::recover_memory();
  const double a = median(t_ad), b = median(t_jac), c = median(t_push),
               d = median(t_pusht);
  const double ea = median(e_ad), eb = median(e_jac), ec = median(e_push),
               ed = median(e_pusht);
  std::cout << std::left << std::setw(10) << name << std::right << std::setw(4)
            << Rhs::N << std::setw(5) << theta.size() + (y0_var ? y0.size() : 0)
            << std::fixed << std::setprecision(3) << std::setw(9) << a * 1e3
            << std::setw(9) << b * 1e3 << std::setw(9) << c * 1e3
            << std::setw(9) << d * 1e3 << std::setprecision(1) << std::setw(7)
            << a / b << std::setw(7) << a / d << std::setw(7) << b / d
            << std::setprecision(2) << std::setw(8) << ea * 1e6 << std::setw(8)
            << eb * 1e6 << std::setw(8) << ec * 1e6 << std::setw(8) << ed * 1e6
            << std::scientific << std::setprecision(1) << std::setw(9)
            << std::max(max_rel_diff(r_ad.grads, r_push.grads, tol),
                        max_rel_diff(r_ad.grads, r_pusht.grads, tol))
            << std::endl;
}

/**
 * Per-evaluation cost of the adjoint backward step: J_y^T lambda and
 * J_theta^T lambda by one nested reverse sweep (the adjoint solver's
 * autodiff path), from the dense user Jacobian, and from the pullback.
 */
template <typename Rhs, typename Jac, typename Pull>
void bench_pullback(const std::string& name, const VectorXd& y,
                    const VectorXd& theta) {
  const int N = y.size(), P = theta.size();
  VectorXd lambda(N);
  for (int i = 0; i < N; ++i) {
    lambda(i) = std::cos(1.0 + i);
  }
  VectorXd g_ad(N + P), g_jac(N + P), g_pull(N + P);
  std::vector<double> e_ad, e_jac, e_pull;
  Eigen::Matrix<var, -1, 1> theta_v = theta.template cast<var>();
  for (int r = 0; r < 2001; ++r) {
    e_ad.push_back(seconds([&] {
      stan::math::nested_rev_autodiff nested;
      Eigen::Matrix<var, -1, 1> y_v = y.template cast<var>();
      Eigen::Matrix<var, -1, 1> th_v = theta.template cast<var>();
      Eigen::Matrix<var, -1, 1> fy = Rhs()(0.5, y_v, nullptr, th_v);
      fy.adj() = lambda;
      stan::math::grad();
      g_ad << y_v.adj(), th_v.adj();
    }));
    e_jac.push_back(seconds([&] {
      const MatrixXd J = Jac()(0.5, y, nullptr, theta);
      g_jac.noalias() = J.transpose() * lambda;
    }));
    e_pull.push_back(
        seconds([&] { g_pull = Pull()(0.5, y, lambda, nullptr, theta); }));
  }
  const double diff
      = ((g_ad - g_pull).array().abs() / g_ad.array().abs().max(1.0))
            .maxCoeff();
  const double a = median(e_ad), b = median(e_jac), c = median(e_pull);
  std::cout << std::left << std::setw(11) << name << std::right << std::setw(4)
            << N << std::setw(5) << P << std::fixed << std::setprecision(2)
            << std::setw(9) << a * 1e6 << std::setw(9) << b * 1e6
            << std::setw(9) << c * 1e6 << std::setprecision(1) << std::setw(7)
            << a / b << std::setw(7) << a / c << std::setw(7) << b / c
            << std::scientific << std::setprecision(1) << std::setw(10) << diff
            << std::endl;
}

/** End-to-end ode_adjoint_tol_ctl gradient: AD, (f, jac), (f, jac, pull). */
template <typename Rhs, typename Jac, typename Pull>
void bench_adjoint(const std::string& name, const VectorXd& y0, bool y0_var,
                   const std::vector<double>& ts, const VectorXd& theta,
                   double budget_s) {
  const double tol = 1e-8;
  const VectorXd atol = VectorXd::Constant(y0.size(), tol);
  auto run = [&](int path) {
    Result res;
    {
      Eigen::Matrix<var, -1, 1> theta_v = theta.template cast<var>();
      Eigen::Matrix<var, -1, 1> y0_v = y0.template cast<var>();
      auto solve = [&](const auto& y0_a) {
        if (path == 0) {
          return stan::math::ode_adjoint_tol_ctl(
              Rhs(), y0_a, 0.0, ts, tol, atol, tol, atol, tol, tol, 100000000,
              150, 1, 2, 2, nullptr, theta_v);
        } else if (path == 1) {
          return stan::math::ode_adjoint_tol_ctl(
              Rhs(), Jac(), y0_a, 0.0, ts, tol, atol, tol, atol, tol, tol,
              100000000, 150, 1, 2, 2, nullptr, theta_v);
        } else {
          return stan::math::ode_adjoint_tol_ctl(
              Rhs(), Jac(), Pull(), y0_a, 0.0, ts, tol, atol, tol, atol, tol,
              tol, 100000000, 150, 1, 2, 2, nullptr, theta_v);
        }
      };
      std::vector<Eigen::Matrix<var, -1, 1>> sol
          = y0_var ? solve(y0_v) : solve(y0);
      var lp = 0;
      for (size_t i = 0; i < sol.size(); ++i) {
        for (int k = 0; k < sol[i].size(); ++k) {
          lp += (1.0 + 0.1 * std::sin(1.0 + i + 7.0 * k)) * sol[i](k);
          res.vals.push_back(sol[i](k).val());
        }
      }
      lp.grad();
      for (int j = 0; j < theta_v.size(); ++j) {
        res.grads.push_back(theta_v(j).adj());
      }
      if (y0_var) {
        for (int j = 0; j < y0_v.size(); ++j) {
          res.grads.push_back(y0_v(j).adj());
        }
      }
    }
    stan::math::recover_memory();
    return res;
  };
  const Result r_ad = run(0), r_jac = run(1), r_pull = run(2);
  const double first = seconds([&] { run(0); }) + seconds([&] { run(1); })
                       + seconds([&] { run(2); });
  const int reps
      = std::max(7, std::min(2000, static_cast<int>(budget_s / first)));
  std::vector<double> t_ad, t_jac, t_pull;
  for (int r = 0; r < reps; ++r) {
    t_ad.push_back(seconds([&] { run(0); }));
    t_jac.push_back(seconds([&] { run(1); }));
    t_pull.push_back(seconds([&] { run(2); }));
  }
  const double a = median(t_ad), b = median(t_jac), c = median(t_pull);
  std::cout << std::left << std::setw(10) << name << std::right << std::setw(4)
            << Rhs::N << std::setw(5) << theta.size() + (y0_var ? y0.size() : 0)
            << std::fixed << std::setprecision(3) << std::setw(9) << a * 1e3
            << std::setw(9) << b * 1e3 << std::setw(9) << c * 1e3
            << std::setprecision(1) << std::setw(7) << a / b << std::setw(7)
            << a / c << std::setw(7) << b / c << std::scientific
            << std::setprecision(1) << std::setw(9)
            << max_rel_diff(r_ad.grads, r_pull.grads, tol) << std::setw(9)
            << max_rel_diff(r_jac.grads, r_pull.grads, tol) << std::endl;
}

std::vector<double> linspace_ts(double a, double b, int n) {
  std::vector<double> ts(n);
  for (int i = 0; i < n; ++i) {
    ts[i] = a + (b - a) * i / (n - 1);
  }
  return ts;
}

VectorXd bump(int n) {
  VectorXd y(n);
  for (int i = 0; i < n; ++i) {
    const double x = static_cast<double>(i) / (n - 1);
    y(i) = 0.1 + 0.8 * std::exp(-std::pow((x - 0.5) / 0.1, 2));
  }
  return y;
}

int main(int argc, char** argv) {
  const double budget = argc > 1 ? std::stod(argv[1]) : 1.0;
  VectorXd pk2_y0(3), pk2_th(5), idr_y0(3), idr_th(7), tr_th(3), rd_th(2);
  pk2_y0 << 100, 0, 0;
  pk2_th << 1.0, 4.0, 2.0, 20.0, 50.0;
  idr_y0 << 100, 0, 10;
  idr_th << 1.0, 2.0, 10.0, 5.0, 0.5, 1.0, 0.8;
  tr_th << 2.0, 3.0, 20.0;
  rd_th << 0.01, 1.0;
  VectorXd tr10_y0 = VectorXd::Zero(12), tr30_y0 = VectorXd::Zero(32);
  tr10_y0(0) = 100;
  tr30_y0(0) = 100;
  const std::vector<double> ts_pk = linspace_ts(0.5, 24, 24);
  const std::vector<double> ts_rd = linspace_ts(0.5, 5, 10);

  std::cout
      << "Forward sensitivities, ode_rk45 (rtol = atol = 1e-6): one "
         "gradient [ms], speedups, coupled RHS per evaluation [us]\n"
      << "model       N sens   AD[ms]  jac[ms] pushR[ms] pushT[ms] AD/jac "
         "AD/pushT jac/pushT rhsAD rhsJac rhsPushR rhsPushT dgrad"
      << std::endl;
  bench_forward<pk2_rhs, pk2_jac, pk2_push, pk2_pusht>("pk2", pk2_y0, false,
                                                       ts_pk, pk2_th, budget);
  bench_forward<pk2_rhs, pk2_jac, pk2_push, pk2_pusht>("pk2 +y0", pk2_y0, true,
                                                       ts_pk, pk2_th, budget);
  bench_forward<idr_rhs, idr_jac, idr_push, idr_pusht>(
      "idr", idr_y0, false, linspace_ts(1, 48, 24), idr_th, budget);
  bench_forward<transit10_rhs, transit10_jac, transit10_push, transit10_pusht>(
      "transit10", tr10_y0, false, ts_pk, tr_th, budget);
  bench_forward<transit30_rhs, transit30_jac, transit30_push, transit30_pusht>(
      "transit30", tr30_y0, false, ts_pk, tr_th, budget);
  bench_forward<rd20_rhs, rd20_jac, rd20_push, rd20_pusht>(
      "rd20", bump(20), false, ts_rd, rd_th, budget);
  bench_forward<rd20_rhs, rd20_jac, rd20_push, rd20_pusht>(
      "rd20 +y0", bump(20), true, ts_rd, rd_th, budget);
  bench_forward<rd50_rhs, rd50_jac, rd50_push, rd50_pusht>(
      "rd50", bump(50), false, ts_rd, rd_th, budget);
  bench_forward<rd50_rhs, rd50_jac, rd50_push, rd50_pusht>(
      "rd50 +y0", bump(50), true, ts_rd, rd_th, budget);

  std::cout << "\nAdjoint backward step per evaluation [us]: J_y^T lambda "
               "and J_theta^T lambda\n"
            << "model        N    P   AD-sweep  jacT*l    pull  AD/jac "
               "AD/pull jac/pull  diff(pull)"
            << std::endl;
  bench_pullback<pk2_rhs, pk2_jac, pk2_pull>("pk2", VectorXd::Constant(3, 0.5),
                                             pk2_th);
  bench_pullback<idr_rhs, idr_jac, idr_pull>("idr", VectorXd::Constant(3, 0.5),
                                             idr_th);
  bench_pullback<transit30_rhs, transit30_jac, transit30_pull>(
      "transit30", VectorXd::Constant(32, 0.5), tr_th);
  bench_pullback<rd20_rhs, rd20_jac, rd20_pull>("rd20", bump(20), rd_th);
  bench_pullback<rd50_rhs, rd50_jac, rd50_pull>("rd50", bump(50), rd_th);
  std::cout << "\nAdjoint end to end, ode_adjoint_tol_ctl (BDF both ways, "
               "tol 1e-8): one gradient [ms]\n"
            << "model       N sens   AD[ms]  jac[ms] pull[ms] AD/jac AD/pull "
               "jac/pull dgrad(AD) dgrad(jac)"
            << std::endl;
  bench_adjoint<pk2_rhs, pk2_jac, pk2_pull>("pk2", pk2_y0, false, ts_pk, pk2_th,
                                            budget);
  bench_adjoint<idr_rhs, idr_jac, idr_pull>(
      "idr", idr_y0, false, linspace_ts(1, 48, 24), idr_th, budget);
  bench_adjoint<transit30_rhs, transit30_jac, transit30_pull>(
      "transit30", tr30_y0, false, ts_pk, tr_th, budget);
  bench_adjoint<rd20_rhs, rd20_jac, rd20_pull>("rd20", bump(20), false, ts_rd,
                                               rd_th, budget);
  bench_adjoint<rd20_rhs, rd20_jac, rd20_pull>("rd20 +y0", bump(20), true,
                                               ts_rd, rd_th, budget);
  bench_adjoint<rd50_rhs, rd50_jac, rd50_pull>("rd50", bump(50), false, ts_rd,
                                               rd_th, budget);
  bench_adjoint<rd50_rhs, rd50_jac, rd50_pull>("rd50 +y0", bump(50), true,
                                               ts_rd, rd_th, budget);
  return 0;
}
