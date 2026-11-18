"""Generate C++ right-hand-side and Jacobian functors for the ODE Jacobian
benchmark from symbolic model definitions (sympy), the way rxode2/nlmixr2
derive sensitivities with symengine.

Each model becomes two stanc3-shaped functors:

- ``<name>_rhs``: ``operator()(t, y, msgs, theta)``, templated on the scalar
  types so the autodiff path can differentiate it.
- ``<name>_jac``: same arguments, returns the N x (N + P) matrix
  [df/dy, df/dtheta] in doubles.

Usage: python3 gen_models.py > models.hpp
"""

import sympy as sp


def logistic_rd(n):
    """1-D reaction-diffusion with logistic growth, zero-flux boundaries."""
    y = sp.symbols(f"y0:{n}")
    D, r = sp.symbols("D r")
    h2 = sp.Rational(1, (n - 1) ** 2)
    f = []
    for i in range(n):
        left = y[i - 1] if i > 0 else y[i + 1]
        right = y[i + 1] if i < n - 1 else y[i - 1]
        f.append(D * (left - 2 * y[i] + right) / h2 + r * y[i] * (1 - y[i]))
    return y, (D, r), f


def transit(n):
    """Transit-compartment absorption into a one-compartment model."""
    a = sp.symbols(f"a0:{n + 2}")
    ktr, cl, v = sp.symbols("ktr CL V")
    f = [-ktr * a[0]]
    for i in range(1, n + 1):
        f.append(ktr * a[i - 1] - ktr * a[i])
    f.append(ktr * a[n] - cl / v * a[n + 1])
    return a, (ktr, cl, v), f


def pk2():
    """Two-compartment PK with first-order absorption."""
    a = sp.symbols("a0:3")
    ka, cl, q, v1, v2 = sp.symbols("ka CL Q V1 V2")
    f = [
        -ka * a[0],
        ka * a[0] - (cl + q) / v1 * a[1] + q / v2 * a[2],
        q / v1 * a[1] - q / v2 * a[2],
    ]
    return a, (ka, cl, q, v1, v2), f


def idr():
    """One-compartment oral PK driving an indirect response (inhibition of
    the production rate kin, Emax model)."""
    a0, a1, r = sp.symbols("a0 a1 R")
    ka, cl, v, kin, kout, ic50, imax = sp.symbols("ka CL V kin kout IC50 Imax")
    c = a1 / v
    f = [
        -ka * a0,
        ka * a0 - cl / v * a1,
        kin * (1 - imax * c / (ic50 + c)) - kout * r,
    ]
    return (a0, a1, r), (ka, cl, v, kin, kout, ic50, imax), f


def robertson():
    """Robertson's stiff chemical kinetics."""
    y = sp.symbols("y0:3")
    k1, k2, k3 = sp.symbols("k1 k2 k3")
    f = [
        -k1 * y[0] + k3 * y[1] * y[2],
        k1 * y[0] - k3 * y[1] * y[2] - k2 * y[1] ** 2,
        k2 * y[1] ** 2,
    ]
    return y, (k1, k2, k3), f


def hires():
    """HIRES stiff test problem (Hairer & Wanner), rate constants as
    parameters."""
    y = sp.symbols("y0:8")
    k = sp.symbols("k0:12")
    (k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12) = k
    f = [
        -k1 * y[0] + k2 * y[1] + k3 * y[2] + k4,
        k1 * y[0] - k5 * y[1],
        -k6 * y[2] + k2 * y[3] + k7 * y[4],
        k3 * y[1] + k1 * y[2] - k8 * y[3],
        -k9 * y[4] + k2 * y[5] + k2 * y[6],
        -k10 * y[5] * y[7] + k11 * y[3] + k1 * y[4] - k2 * y[5] + k11 * y[6],
        k10 * y[5] * y[7] - k12 * y[6],
        -k10 * y[5] * y[7] + k12 * y[6],
    ]
    return y, k, f


MODELS = {
    "pk2": pk2(),
    "idr": idr(),
    "transit10": transit(10),
    "transit30": transit(30),
    "robertson": robertson(),
    "hires": hires(),
    "rd20": logistic_rd(20),
    "rd50": logistic_rd(50),
}


def cxx(expr, subs):
    # unqualified pow: argument-dependent lookup picks stan::math::pow for
    # autodiff scalars, the file-level using-declaration std::pow for doubles
    code = sp.cxxcode(expr.xreplace(subs), standard="c++17")
    return code.replace("std::pow(", "pow(")


def emit(name, states, params, f):
    n, p = len(states), len(params)
    subs = {s: sp.Symbol(f"y({i})") for i, s in enumerate(states)}
    subs.update({s: sp.Symbol(f"theta({j})") for j, s in enumerate(params)})
    out = []
    # right-hand side (templated, differentiated by the autodiff path)
    out.append(f"struct {name}_rhs {{")
    out.append(f"  static constexpr int N = {n};")
    out.append(f"  static constexpr int P = {p};")
    out.append("  template <typename T0, typename TY, typename TP>")
    out.append("  Eigen::Matrix<stan::return_type_t<T0, TY, TP>, -1, 1>")
    out.append("  operator()(const T0& t, const TY& y, std::ostream* msgs,")
    out.append("             const TP& theta) const {")
    out.append("    using S = stan::return_type_t<T0, TY, TP>;")
    reps, red = sp.cse(f)
    for sym, e in reps:
        out.append(f"    const S {sym} = {cxx(e, subs)};")
    out.append(f"    Eigen::Matrix<S, -1, 1> dy({n});")
    for i, e in enumerate(red):
        out.append(f"    dy({i}) = {cxx(e, subs)};")
    out.append("    return dy;")
    out.append("  }")
    out.append("};")
    # Jacobian [df/dy, df/dtheta]
    entries = []
    for i, fi in enumerate(f):
        for j, v in enumerate(list(states) + list(params)):
            d = sp.diff(fi, v)
            if d != 0:
                entries.append(((i, j), d))
    reps, red = sp.cse([e for _, e in entries])
    out.append(f"struct {name}_jac {{")
    out.append("  template <typename T0, typename TY, typename TP>")
    out.append("  Eigen::Matrix<stan::return_type_t<T0, TY, TP>, -1, -1>")
    out.append("  operator()(const T0& t, const TY& y, std::ostream* msgs,")
    out.append("             const TP& theta) const {")
    out.append("    using S = stan::return_type_t<T0, TY, TP>;")
    for sym, e in reps:
        out.append(f"    const S {sym} = {cxx(e, subs)};")
    out.append(
        f"    Eigen::Matrix<S, -1, -1> J = Eigen::Matrix<S, -1, -1>::Zero({n}, {n + p});"
    )
    for ((i, j), _), e in zip(entries, red):
        out.append(f"    J({i}, {j}) = {cxx(e, subs)};")
    out.append("    return J;")
    out.append("  }")
    out.append("};")
    return "\n".join(out)


def main():
    print("// Generated by gen_models.py; do not edit.")
    print("#ifndef ODE_JACOBIAN_BENCHMARK_MODELS_HPP")
    print("#define ODE_JACOBIAN_BENCHMARK_MODELS_HPP")
    print("#include <stan/math.hpp>")
    print("#include <cmath>")
    print("using std::pow;")
    for name, (states, params, f) in MODELS.items():
        print(emit(name, states, params, f))
    print("#endif")


if __name__ == "__main__":
    main()
