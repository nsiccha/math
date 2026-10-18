# Reference solution of the delayed indirect-response model of
# pmx_solve_dde_rk45_test.cpp (test delayed_indirect_response_reference),
# computed independently with DelayDiffEq.jl (method of steps over Vern9).
#
#   julia --project=<env with DelayDiffEq, OrdinaryDiffEqVerner> pmx_solve_dde_rk45_reference.jl
#
# prints the amounts at the observation times as a C++ initializer list.

using DelayDiffEq, OrdinaryDiffEqVerner
const SciMLBase = DelayDiffEq.SciMLBase

# theta = (ka, CL, V, kin, kout, Smax, SC50); two delays; history y_hist
const THETA = [0.9, 4.0, 30.0, 2.0, 0.25, 1.5, 2.0]
const TAU = [2.0, 5.0]
const Y_HIST = [0.0, 0.0, 7.0]

# oral doses into compartment 1 (t = 0, then ADDL = 3 every II = 24), an IV
# bolus into compartment 2, and an infusion into compartment 2
const ORAL_TIMES = [0.0, 24.0, 48.0, 72.0]
const ORAL_DOSE = 100.0
const IV_TIME = 30.0
const IV_DOSE = 40.0
const INFUSION_START = 55.0
const INFUSION_RATE = 20.0
const INFUSION_END = INFUSION_START + 60.0 / INFUSION_RATE
const OBS_TIMES = [1.5, 3.3, 7.1, 11.7, 18.2, 26.9, 31.4, 36.6, 44.2, 50.8, 57.3,
                   63.9, 75.5, 88.1, 99.4]

# the infusion rate is switched by callbacks, so that a step ending at a switch
# evaluates the right hand side with the rate before it
const RATE = Ref(0.0)

function rhs!(du, u, h, p, t)
    ka, CL, V, kin, kout, Smax, SC50 = p
    C_lag = h(p, t - TAU[1]; idxs=2) / V
    R_lag = h(p, t - TAU[2]; idxs=3)
    du[1] = -ka * u[1]
    du[2] = ka * u[1] - CL / V * u[2] + RATE[]
    du[3] = kin * (1 + Smax * C_lag / (SC50 + C_lag)) * sqrt(kin / kout / R_lag) - kout * u[3]
    nothing
end

history(p, t; idxs=nothing) = idxs === nothing ? Y_HIST : Y_HIST[idxs]

function main()
    u0 = Y_HIST + [ORAL_DOSE, 0.0, 0.0]
    events = sort(vcat(ORAL_TIMES[2:end], IV_TIME, INFUSION_START, INFUSION_END))
    # every event is a discontinuity of the state or of its derivative, and
    # propagates through the delays
    sources = vcat(0.0, events)
    propagated = [u + k1 * TAU[1] + k2 * TAU[2] for u in sources
                  for k1 in 0:5 for k2 in 0:5 if 1 <= k1 + k2 <= 5]
    tmax = maximum(OBS_TIMES)
    stops = sort(unique(filter(t -> 0 < t <= tmax, vcat(events, propagated))))
    affect_oral!(integrator) = (integrator.u[1] += ORAL_DOSE)
    affect_iv!(integrator) = (integrator.u[2] += IV_DOSE)
    # discrete callbacks at the event times, which are all in tstops
    at(times, affect!) = SciMLBase.DiscreteCallback((u, t, integrator) -> t in times, affect!;
                                                    save_positions=(false, false))
    callbacks = SciMLBase.CallbackSet(
        at(ORAL_TIMES[2:end], affect_oral!),
        at([IV_TIME], affect_iv!),
        at([INFUSION_START], integrator -> (RATE[] = INFUSION_RATE)),
        at([INFUSION_END], integrator -> (RATE[] = 0.0)))
    RATE[] = 0.0
    prob = DDEProblem(rhs!, u0, history, (0.0, tmax), THETA; constant_lags=TAU)
    sol = solve(prob, MethodOfSteps(Vern9()); reltol=1e-13, abstol=1e-13,
                saveat=OBS_TIMES, save_start=false, tstops=stops,
                d_discontinuities=stops, callback=callbacks, maxiters=10^7)
    SciMLBase.successful_retcode(sol) || error("solve failed: $(sol.retcode)")
    sol.t == OBS_TIMES || error("saved times $(sol.t) differ from $(OBS_TIMES)")
    for (t, u) in zip(sol.t, sol.u)
        println("    ", join((repr(x) for x in u), ", "), ",  // t = ", t)
    end
end

main()
