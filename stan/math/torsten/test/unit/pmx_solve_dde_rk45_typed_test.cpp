#include <stan/math/torsten/test/unit/test_fixture_onecpt.hpp>
#include <stan/math/torsten/test/unit/test_fixture_twocpt.hpp>
#include <stan/math/torsten/test/unit/test_functors.hpp>
#include <stan/math/torsten/pmx_onecpt_model.hpp>
#include <stan/math/torsten/pmx_twocpt_model.hpp>
#include <stan/math/torsten/pmx_solve_dde_rk45.hpp>
#include <boost/mp11.hpp>

/*
 * pmx_solve_dde_rk45 with a right hand side that ignores its lagged states
 * (and a zero history) solves an ODE, so on Torsten's ODE test schedules it
 * must reproduce pmx_solve_rk45 (solver 2), values and gradients. Solver 1
 * is the DDE solver, so compare_val and the overload and finite-difference
 * checks of the fixture apply to it.
 */

using var = stan::math::var_value<double>;
using torsten::PMXOneCptODE;
using torsten::PMXTwoCptODE;

TYPED_TEST_SUITE_P(test_onecpt);

TYPED_TEST_P(test_onecpt, multiple_bolus) {
  Eigen::MatrixXd amounts(10, 2);
  amounts << 1000.0, 0.0,
    740.8182, 254.97490,
    548.8116, 436.02020,
    406.5697, 562.53846,
    301.1942, 648.89603,
    223.1302, 705.72856,
    165.2989, 740.90816,
    122.4564, 760.25988,
    90.71795, 768.09246,
    8.229747, 667.87079;
  Eigen::MatrixXd x = amounts.transpose();
  this -> compare_val(x, 5.e-5);
  this -> compare_solvers_val();

  this -> biovar[0] = {0.8, 0.9};
  this -> compare_solvers_val();
  this -> compare_solvers_adj(this -> amt, 1.e-8, "AMT");
  this -> compare_solvers_adj(this -> rate, 1.e-8, "RATE");
  this -> compare_solvers_adj(this -> ii, 1.e-8, "II");
  this -> compare_solvers_adj(this -> theta[0], 5.e-6, "theta");
  this -> compare_solvers_adj(this -> biovar[0], 1.e-6, "bioavailability");
  this -> compare_solvers_adj(this -> tlag[0], 1.e-8, "lag time");
}

TYPED_TEST_P(test_onecpt, multiple_bolus_cent) {
  this -> cmt[0] = 2;
  this -> biovar[0] = {0.8, 0.9};
  this -> compare_solvers_val();
  this -> compare_solvers_adj(this -> amt, 1.e-8, "AMT");
  this -> compare_solvers_adj(this -> theta[0], 5.e-6, "theta");
  this -> compare_solvers_adj(this -> biovar[0], 1.e-6, "bioavailability");
}

TYPED_TEST_P(test_onecpt, multiple_bolus_with_tlag) {
  // the lagged doses (4.6, 5.9, 7.2, 8.5) fall between observations, so
  // the solution is smooth in the lag time and the interval
  this -> tlag[0][0] = 4.6;  // tlag1
  this -> tlag[0][1] = 0;  // tlag2
  this -> time[0] = 0;
  for(int i = 1; i < 10; ++i) this -> time[i] = this -> time[i - 1] + 1;
  this -> amt[0] = 1200;
  this -> ii[0] = 1.3;
  this -> addl[0] = 3;

  this -> compare_solvers_val();
  this -> compare_solvers_adj(this -> amt, 1.e-8, "AMT");
  this -> compare_solvers_adj(this -> ii, 1.e-6, "II");
  this -> compare_solvers_adj(this -> theta[0], 5.e-6, "theta");
  this -> compare_solvers_adj(this -> tlag[0], 1.e-6, "lag time");
  this -> test_finite_diff_tlag(1.e-3, 1.e-3);
  this -> test_finite_diff_ii(1.e-3, 1.e-2);
}

TYPED_TEST_P(test_onecpt, single_infusion) {
  this -> amt[0] = 1200;
  this -> rate[0] = 1200;
  this -> addl[0] = 0;

  Eigen::MatrixXd amounts(10, 2);
  amounts << 0.00000,   0.00000,
             259.18178,  40.38605,
             451.18836, 145.61440,
             593.43034, 296.56207,
             698.80579, 479.13371,
             517.68806, 642.57025,
             383.51275, 754.79790,
             284.11323, 829.36134,
             210.47626, 876.28631,
             19.09398, 844.11769;
  Eigen::MatrixXd x = amounts.transpose();
  this -> compare_val(x);
  this -> compare_solvers_val();

  this -> biovar[0] = {0.7, 1.0};
  this -> compare_solvers_val();
  this -> compare_solvers_adj(this -> amt, 1.e-6, "AMT");
  this -> compare_solvers_adj(this -> rate, 1.e-6, "RATE");
  this -> compare_solvers_adj(this -> theta[0], 1.5e-6, "theta");
  this -> compare_solvers_adj(this -> biovar[0], 1.e-6, "bioavailability");

  // the infusion (0.92 h) ends between observations, so the solution is
  // smooth in its amount and rate
  this -> amt[0] = 1100;
  this -> test_finite_diff_amt(1.0, 1.e-6);
  this -> test_finite_diff_rate(1.e-1, 1.e-4);
}

TYPED_TEST_P(test_onecpt, single_infusion_cent) {
  this -> amt[0] = 1000;
  this -> rate[0] = 600;
  this -> cmt[0] = 2;

  Eigen::MatrixXd amounts(10, 2);
  amounts <<
    0,           0,
    0, 147.6804745,
    0, 290.8172984,
    0, 429.5502653,
    0, 564.0148675,
    0, 694.3424289,
    0, 820.6602327,
    0, 893.3511610,
    0,  865.865635,
    0, 674.3368348;
  Eigen::MatrixXd x = amounts.transpose();
  this -> compare_val(x, 1.e-6);

  this -> amt[0] = 1200;
  this -> rate[0] = 1200;
  this -> addl[0] = 0;
  this -> compare_solvers_val();
  this -> compare_solvers_adj(this -> amt, 1.e-6, "AMT");
  this -> compare_solvers_adj(this -> rate, 1.e-6, "RATE");
  this -> compare_solvers_adj(this -> theta[0], 1.e-6, "theta");
}

TYPED_TEST_P(test_onecpt, multiple_infusion_addl) {
  this -> time[0] = 0.0;
  for(int i = 1; i < this -> nt; i++) this -> time[i] = this -> time[i - 1] + 1.7;
  this -> amt[0] = 1200;
  this -> rate[0] = 500;
  this -> ii[0] = 4;
  this -> addl[0] = 3;
  this -> biovar[0] = {0.8, 1.0};

  this -> compare_solvers_val();
  this -> compare_solvers_adj(this -> amt, 1.e-6, "AMT");
  this -> compare_solvers_adj(this -> rate, 1.e-6, "RATE");
  this -> compare_solvers_adj(this -> ii, 1.e-6, "II");
  this -> compare_solvers_adj(this -> theta[0], 2.e-6, "theta");
  this -> compare_solvers_adj(this -> biovar[0], 1.e-6, "bioavailability");
  this -> test_finite_diff_ii(1.e-3, 1.e-2);
}

TYPED_TEST_P(test_onecpt, time_dependent_theta) {
  this -> reset_events(11);
  int nt = this -> nt;

  this -> theta.resize(nt);
  for (int i = 0; i < nt; i++) {
    this -> theta[i].resize(3);
    this -> theta[i][0] = i < 6 ? 10 : 50;  // CL is piece-wise constant
    this -> theta[i][1] = 80;              // Vc
    this -> theta[i][2] = 1.2;             // ka
  }

  this -> time[0] = 0.0;
  for(int i = 1; i < nt; i++) {
    this -> time[i] = this -> time[i - 1] + 2.5;
  }

  this -> amt[0] = 1000;
  this -> cmt[0] = 1;
  this -> evid[0] = 1;
  this -> ii[0] = 12;
  this -> addl[0] = 1;

  Eigen::MatrixXd amounts(nt, 2);
  amounts << 1000.0, 0.0,
    4.978707e+01, 761.1109513,
    2.478752e+00, 594.7341503,
    1.234098e-01, 437.0034049,
    6.144212e-03, 319.8124495,
    5.488119e+02, 670.0046601,
    2.732374e+01, 323.4948561,
    1.360369e+00, 76.9219400,
    6.772877e-02, 16.5774607,
    3.372017e-03, 3.4974152,
    1.678828e-04, 0.7342228;
  Eigen::MatrixXd x = amounts.transpose();
  this -> compare_val(x, 5.e-5);

  for (int i = 0; i < nt; ++i) {
    this -> compare_solvers_adj(this -> theta[i], 5.e-6, "theta");
  }
}

TYPED_TEST_P(test_onecpt, reset) {
  this -> evid[4] = 3;         // reset
  this -> evid[6] = 4;         // reset and dose
  this -> amt[6] = 500;
  this -> cmt[6] = 2;
  this -> addl[0] = 0;

  this -> compare_solvers_val();
  this -> compare_solvers_adj(this -> amt, 1.e-8, "AMT");
  this -> compare_solvers_adj(this -> theta[0], 5.e-6, "theta");
}

TYPED_TEST_P(test_onecpt, turn_off_cmt) {
  this -> evid[4] = 2;         // turn off the central compartment
  this -> cmt[4] = -2;
  this -> evid[6] = 1;
  this -> amt[6] = 500;
  this -> cmt[6] = 1;
  this -> addl[0] = 0;

  this -> compare_solvers_val();
  this -> compare_solvers_adj(this -> amt, 1.e-8, "AMT");
  this -> compare_solvers_adj(this -> theta[0], 5.e-6, "theta");
}

TYPED_TEST_P(test_onecpt, overloads) {
  this -> compare_overload(1.e-4);
}

TYPED_TEST_P(test_onecpt, multiple_bolus_ss) {
  // burn-in to the steady state converged below the comparisons (and
  // within reach of pmx_solve_rk45's Newton solve)
  this -> as_rtol = 1.e-8;
  this -> as_atol = 1.e-8;
  this -> as_max_num_steps = 1000;
  this -> time[0] = 0.0;
  this -> time[1] = 0.0;
  for(int i = 2; i < 10; i++) this -> time[i] = this -> time[i - 1] + 5;
  this -> amt[0] = 1200;
  this -> addl[0] = 10;
  this -> ss[0] = 1;

  Eigen::MatrixXd amounts(10, 2);
  amounts << 1200.0, 384.7363,
    1200.0, 384.7363,
    2.974504, 919.6159,
    7.373062e-3, 494.0040,
    3.278849e+1, 1148.4725,
    8.127454e-2, 634.2335,
    3.614333e+2, 1118.2043,
    8.959035e-1, 813.4883,
    2.220724e-3, 435.9617,
    9.875702, 1034.7998;
  Eigen::MatrixXd x = amounts.transpose();
  this -> compare_rel_val(x, 1.e-6);

  this -> biovar[0] = {0.8, 0.9};
  this -> compare_solvers_val(1.e-5);
  this -> compare_solvers_adj(this -> amt, 1.e-6, "AMT");
  this -> compare_solvers_adj(this -> theta[0], 5.e-6, "theta");
  this -> compare_solvers_adj(this -> biovar[0], 1.e-6, "bioavailability");
  this -> compare_solvers_adj(this -> ii, 1.e-6, "II");

  this -> tlag[0][0] = 1.7;
  this -> compare_solvers_val(1.e-5);
  this -> compare_solvers_adj(this -> tlag[0], 5.e-7, "lag time");
}

TYPED_TEST_P(test_onecpt, multiple_infusion_ss) {
  this -> as_rtol = 1.e-8;
  this -> as_atol = 1.e-8;
  this -> as_max_num_steps = 1000;
  this -> time[0] = 0.0;
  this -> time[1] = 0.0;
  for(int i = 2; i < 10; i++) this -> time[i] = this -> time[i - 1] + 5;
  this -> amt[0] = 1200;
  this -> rate[0] = 150;
  this -> addl[0] = 10;
  this -> ss[0] = 1;

  Eigen::MatrixXd amounts(10, 2);
  amounts << 1.028649, 659.9385,
    1.028649, 659.9385,
    124.692706, 837.1959,
    11.338982, 836.1947,
    121.612641, 737.4911,
    124.991604, 950.4222,
    87.660547, 642.9529,
    124.907445, 879.6271,
    3.415236, 745.2971,
    123.979747, 789.6393;
  Eigen::MatrixXd x = amounts.transpose();
  this -> compare_val(x, 5.e-5);
  this -> compare_solvers_val(1.e-5);
  this -> compare_solvers_adj(this -> amt, 1.e-6, "AMT");
  this -> compare_solvers_adj(this -> rate, 1.e-6, "RATE");
  this -> compare_solvers_adj(this -> theta[0], 5.e-6, "theta");
  this -> compare_solvers_adj(this -> ii, 1.e-5, "II");

  this -> tlag[0] = {2.4, 1.7};
  this -> compare_solvers_val(1.e-5);
  this -> compare_solvers_adj(this -> amt, 1.e-6, "AMT");
  this -> compare_solvers_adj(this -> rate, 1.e-6, "RATE");
  this -> compare_solvers_adj(this -> tlag[0], 5.e-7, "lag time");
}

TYPED_TEST_P(test_onecpt, constant_infusion_ss) {
  this -> time[0] = 0.0;
  for(int i = 1; i < this -> nt; i++) this -> time[i] = this -> time[i - 1] + 2.5;
  this -> amt[0] = 0.0;
  this -> rate[0] = 150;
  this -> ii[0] = 0.0;
  this -> addl[0] = 0;
  this -> ss[0] = 1;

  this -> compare_solvers_val(1.e-5);
  this -> compare_solvers_adj(this -> rate, 1.e-6, "RATE");
  this -> compare_solvers_adj(this -> theta[0], 1.e-6, "theta");
}

REGISTER_TYPED_TEST_SUITE_P(test_onecpt,
                            multiple_bolus,
                            multiple_bolus_cent,
                            multiple_bolus_with_tlag,
                            single_infusion,
                            single_infusion_cent,
                            multiple_infusion_addl,
                            time_dependent_theta,
                            reset,
                            turn_off_cmt,
                            overloads,
                            multiple_bolus_ss,
                            multiple_infusion_ss,
                            constant_infusion_ss);

using onecpt_test_types = ::testing::Types<
  std::tuple<pmx_solve_dde_rk45_lag_free_functor, pmx_solve_rk45_functor,
             double, double, double, double, double, double, double, PMXOneCptODE>,
  std::tuple<pmx_solve_dde_rk45_lag_free_functor, pmx_solve_rk45_functor,
             double, var, var, var, var, var, var, PMXOneCptODE> >;

INSTANTIATE_TYPED_TEST_SUITE_P(PMX_DDE, test_onecpt, onecpt_test_types);

TYPED_TEST_SUITE_P(test_twocpt);

TYPED_TEST_P(test_twocpt, multiple_bolus) {
  this -> biovar[0] = {0.8, 0.9, 0.7};
  this -> compare_solvers_val();
  this -> compare_solvers_adj(this -> amt, 1.e-8, "AMT");
  this -> compare_solvers_adj(this -> ii, 1.e-8, "II");
  this -> compare_solvers_adj(this -> theta[0], 5.e-6, "theta");
  this -> compare_solvers_adj(this -> biovar[0], 1.e-6, "bioavailability");
}

TYPED_TEST_P(test_twocpt, time_dependent_theta) {
  this -> reset_events(11);
  this -> theta.resize(this -> nt);
  for (int i = 0; i < this -> nt; i++) {
    this -> theta[i].resize(5);
    this -> theta[i][0] = i < 6 ? 5 : 50;  // CL is piece-wise constant
    this -> theta[i][1] = 8;  // Q
    this -> theta[i][2] = 20;  // Vc
    this -> theta[i][3] = 70;  // Vp
    this -> theta[i][4] = 1.2;  // ka
  }

  this -> time[0] = 0.0;
  for(int i = 1; i < this -> nt; i++) this -> time[i] = this -> time[i - 1] + 2.5;

  this -> amt[0] = 1000;
  this -> cmt[0] = 1;
  this -> evid[0] = 1;
  this -> ii[0] = 12;
  this -> addl[0] = 1;

  Eigen::MatrixXd amounts(this -> nt, 3);
  amounts << 1.000000e+03,   0.000000,   0.0000,
    4.978707e+01, 352.089056, 349.4148,
    2.478752e+00, 146.871246, 458.3010,
    1.234098e-01,  93.537648, 442.6420,
    6.144212e-03,  77.732083, 405.7800,
    5.488119e+02, 449.105589, 412.0337,
    2.732374e+01,  36.675537, 430.0023,
    1.360369e+00,  14.886990, 341.6754,
    6.772877e-02,  10.966107, 267.7033,
    3.372017e-03,   8.549649, 209.5604,
    1.678828e-04,   6.690631, 164.0364;
  Eigen::MatrixXd x = amounts.transpose();
  this -> compare_val(x, 5.e-5);

  for (auto i = 0; i < this -> nt; ++i) {
    this -> compare_solvers_adj(this -> theta[i], 5.e-6, "theta");
  }
}

TYPED_TEST_P(test_twocpt, multiple_infusion_2) {
  this -> theta[0][0] = 5;  // CL
  this -> theta[0][1] = 8;  // Q
  this -> theta[0][2] = 35;  // Vc
  this -> theta[0][3] = 105;  // Vp
  this -> theta[0][4] = 1.2;  // ka
  this -> amt[0] = 1200;
  this -> rate[0] = 1200;

  Eigen::MatrixXd amounts(10, 3);
  amounts << 0.00000,   0.00000,   0.0000000,
             259.18178,  39.55748,   0.7743944,
             451.18836, 139.65573,   5.6130073,
             593.43034, 278.43884,  17.2109885,
             698.80579, 440.32663,  37.1629388,
             517.68806, 574.76950,  65.5141658,
             383.51275, 653.13596,  99.2568509,
             284.11323, 692.06145, 135.6122367,
             210.47626, 703.65965, 172.6607082,
             19.09398, 486.11014, 406.6342765;
  Eigen::MatrixXd x = amounts.transpose();
  this -> compare_val(x);

  std::fill(this -> rate.begin(), this -> rate.end(), 340);
  std::fill(this -> amt.begin(), this -> amt.end(), 1000);
  std::fill(this -> evid.begin(), this -> evid.end(), 1);
  this -> compare_solvers_val();
  this -> compare_solvers_adj(this -> amt, 1.e-6, "AMT");
  this -> compare_solvers_adj(this -> rate, 1.e-6, "RATE");
  this -> compare_solvers_adj(this -> theta[0], 2.e-6, "theta");
}

REGISTER_TYPED_TEST_SUITE_P(test_twocpt,
                            multiple_bolus,
                            time_dependent_theta,
                            multiple_infusion_2);

using twocpt_test_types = ::testing::Types<
  std::tuple<pmx_solve_dde_rk45_lag_free_functor, pmx_solve_rk45_functor,
             double, double, double, double, double, double, double, PMXTwoCptODE>,
  std::tuple<pmx_solve_dde_rk45_lag_free_functor, pmx_solve_rk45_functor,
             double, var, var, var, var, var, var, PMXTwoCptODE> >;

INSTANTIATE_TYPED_TEST_SUITE_P(PMX_DDE, test_twocpt, twocpt_test_types);
