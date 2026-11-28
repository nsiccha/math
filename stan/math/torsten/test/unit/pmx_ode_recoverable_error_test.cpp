#include <stan/math.hpp>
#include <stan/math/rev/core.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

using stan::math::var;

/*
 * An ODE or steady-state solve that fails for the parameter values it was
 * given must throw std::domain_error. Since Stan 2.35 the samplers reject
 * the proposal on a std::domain_error and stop the chain on any other
 * exception (through Stan 2.34 they rejected on any std::exception). The
 * sampler reaches such values routinely early in warmup, so a solver that
 * reports them as std::runtime_error ends the chain there:
 *
 * - the rk45 integrator let boost::odeint's step-budget error
 *   ("Max number of iterations exceeded (1000000).", a std::runtime_error)
 *   escape, where stan::math::ode_rk45 and the CVODES integrators throw
 *   std::domain_error;
 * - the steady-state solves of the ODE and coupled models rethrew every
 *   failure, a std::domain_error of the ODE or algebra solver included,
 *   as std::runtime_error.
 */

namespace {

/* The right-hand side the Stan compiler generates for a one-compartment
 * model with first-order absorption: depot, central. */
struct depot_1cmt_ode {
  template <typename T0, typename T1, typename T2, typename T3>
  Eigen::Matrix<stan::return_type_t<T0, T1, T2, T3>, -1, 1> operator()(
      const T0& t, const Eigen::Matrix<T1, -1, 1>& y,
      const std::vector<T2>& params, const std::vector<T3>& x_r,
      const std::vector<int>& x_i, std::ostream* pstream__) const {
    using scalar = stan::return_type_t<T0, T1, T2, T3>;
    scalar cl = params[0], vc = params[1], ka = params[2];
    Eigen::Matrix<scalar, -1, 1> dydt(2);
    dydt[0] = -ka * y[0];
    dydt[1] = ka * y[0] - cl / vc * y[1];
    return dydt;
  }
};

/* One effect state driven by the central concentration of the
 * one-compartment PK model, for the coupled solvers. */
struct effect_ode {
  template <typename T0, typename T1, typename T2, typename T3>
  Eigen::Matrix<stan::return_type_t<T0, T1, T2, T3>, -1, 1> operator()(
      const T0& t, const Eigen::Matrix<T1, -1, 1>& x,
      const Eigen::Matrix<T2, -1, 1>& x_pk, const std::vector<T3>& theta,
      const std::vector<double>& x_r, const std::vector<int>& x_i,
      std::ostream* pstream__) const {
    using scalar = stan::return_type_t<T0, T1, T2, T3>;
    Eigen::Matrix<scalar, -1, 1> dxdt(1);
    dxdt[0] = theta[3] * (x_pk[1] / theta[1] - x[0]);
    return dxdt;
  }
};

/* Runs a solve that is expected to fail and says how it failed. */
template <typename F>
::testing::AssertionResult throws_domain_error(const F& run,
                                               const std::string& expected) {
  ::testing::AssertionResult result = ::testing::AssertionFailure()
                                      << "did not throw";
  try {
    run();
  } catch (const std::domain_error& e) {
    if (std::string(e.what()).find(expected) == std::string::npos) {
      result = ::testing::AssertionFailure()
               << "threw a std::domain_error without \"" << expected
               << "\": " << e.what();
    } else {
      result = ::testing::AssertionSuccess();
    }
  } catch (const std::exception& e) {
    result = ::testing::AssertionFailure()
             << "threw " << typeid(e).name()
             << ", which is not a std::domain_error: " << e.what();
  }
  stan::math::recover_memory();
  return result;
}

struct TorstenOdeRecoverableError : public ::testing::Test {
  /* Doses of 200 into the depot every 24 h and central observations. */
  std::vector<double> time{0, 0.25, 0.5, 1, 2, 4, 8, 12, 24, 168};
  std::vector<double> amt, rate, ii;
  std::vector<int> evid, cmt, addl, ss;
  std::vector<double> biovar{1.0, 1.0}, tlag{0.0, 0.0};
  const depot_1cmt_ode f{};
  const std::vector<double> truth{0.5, 12.0, 1.0};  // CL, V, ka

  TorstenOdeRecoverableError()
      : amt(time.size(), 0.0),
        rate(time.size(), 0.0),
        ii(time.size(), 0.0),
        evid(time.size(), 0),
        cmt(time.size(), 2),
        addl(time.size(), 0),
        ss(time.size(), 0) {
    amt[0] = 200;
    evid[0] = 1;
    cmt[0] = 1;
    ii[0] = 24;
    addl[0] = 6;
  }

  /* The first record becomes one steady-state dose. */
  void steady_state() {
    ss[0] = 1;
    addl[0] = 0;
  }

  template <typename T>
  static std::vector<T> params(const std::vector<double>& x) {
    return std::vector<T>(x.begin(), x.end());
  }
};

TEST_F(TorstenOdeRecoverableError, rk45_step_budget) {
  const double tol = 1e-10;
  const long int steps = 20;  // far too few for this tolerance
  EXPECT_TRUE(throws_domain_error(
      [&] {
        torsten::pmx_solve_rk45(f, 2, time, amt, rate, ii, evid, cmt, addl, ss,
                                params<var>(truth), biovar, tlag, tol, tol,
                                steps, nullptr);
      },
      "Failed to integrate to next output time within 20 steps"));
  EXPECT_TRUE(throws_domain_error(
      [&] {
        torsten::pmx_solve_rk45(f, 2, time, amt, rate, ii, evid, cmt, addl, ss,
                                params<double>(truth), biovar, tlag, tol, tol,
                                steps, nullptr);
      },
      "Failed to integrate to next output time within 20 steps"));
}

/* What a sampler meets early in warmup: an absorption rate so high that
 * the system is stiff, solved with the default controls. The bdf solver
 * handles these values, so they are valid. */
TEST_F(TorstenOdeRecoverableError, rk45_default_controls_on_a_stiff_system) {
  const std::vector<double> stiff{0.5, 12.0, 1e9};
  EXPECT_TRUE(throws_domain_error(
      [&] {
        torsten::pmx_solve_rk45(f, 2, time, amt, rate, ii, evid, cmt, addl, ss,
                                params<var>(stiff), biovar, tlag, nullptr);
      },
      "Failed to integrate to next output time within 1000000 steps"));

  auto x = torsten::pmx_solve_bdf(f, 2, time, amt, rate, ii, evid, cmt, addl,
                                  ss, params<var>(stiff), biovar, tlag,
                                  nullptr);
  for (int i = 0; i < x.cols(); ++i) {
    EXPECT_TRUE(std::isfinite(x(1, i).val()));
  }
  stan::math::recover_memory();
}

TEST_F(TorstenOdeRecoverableError, rk45_step_budget_in_steady_state) {
  steady_state();
  const double tol = 1e-10;
  const long int steps = 20;
  EXPECT_TRUE(throws_domain_error(
      [&] {
        torsten::pmx_solve_rk45(f, 2, time, amt, rate, ii, evid, cmt, addl, ss,
                                params<var>(truth), biovar, tlag, tol, tol,
                                steps, 1e-6, 1e-6, 100, nullptr);
      },
      "Failed to find steady state"));
}

/* One Newton iteration cannot meet these tolerances, so the algebra
 * solver gives up, whichever ODE integrator it calls. */
TEST_F(TorstenOdeRecoverableError, steady_state_solver_gives_up) {
  steady_state();
  const double tol = 1e-6, as_tol = 1e-14;
  const long int steps = 1000000, as_steps = 1;
  EXPECT_TRUE(throws_domain_error(
      [&] {
        torsten::pmx_solve_rk45(f, 2, time, amt, rate, ii, evid, cmt, addl, ss,
                                params<var>(truth), biovar, tlag, tol, tol,
                                steps, as_tol, as_tol, as_steps, nullptr);
      },
      "Failed to find steady state"));
  EXPECT_TRUE(throws_domain_error(
      [&] {
        torsten::pmx_solve_bdf(f, 2, time, amt, rate, ii, evid, cmt, addl, ss,
                               params<var>(truth), biovar, tlag, tol, tol,
                               steps, as_tol, as_tol, as_steps, nullptr);
      },
      "Failed to find steady state"));
}

TEST_F(TorstenOdeRecoverableError, coupled_steady_state_solver_gives_up) {
  steady_state();
  cmt.assign(time.size(), 3);
  cmt[0] = 1;
  const double tol = 1e-6, as_tol = 1e-14;
  const long int steps = 1000000, as_steps = 1;
  const std::vector<std::vector<var>> theta{{0.5, 12.0, 1.0, 0.3}};
  const std::vector<std::vector<double>> biovar2{{1.0, 1.0, 1.0}};
  const std::vector<std::vector<double>> tlag2{{0.0, 0.0, 0.0}};
  const effect_ode f_effect{};
  EXPECT_TRUE(throws_domain_error(
      [&] {
        torsten::pmx_solve_onecpt_rk45(f_effect, 1, time, amt, rate, ii, evid,
                                       cmt, addl, ss, theta, biovar2, tlag2,
                                       tol, tol, steps, as_tol, as_tol,
                                       as_steps, nullptr);
      },
      "failed to find steady state"));
}

}  // namespace
