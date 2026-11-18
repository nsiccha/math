#!/usr/bin/env bash
# Build the ODE Jacobian benchmark against this Stan Math checkout.
# Needs lib/tbb and the SUNDIALS static libraries built (make lib/tbb/libtbb.so.2
# lib/sundials_6.1.1/lib/libsundials_cvodes.a ...).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
python3 "$here/gen_models.py" > "$here/models.hpp"
g++ -std=c++17 -O3 -pthread -D_REENTRANT -DBOOST_DISABLE_ASSERTS \
  -Wno-ignored-attributes -Wno-deprecated-declarations \
  -I "$root" -isystem "$root/lib/tbb_2020.3/include" \
  -isystem "$root/lib/eigen_3.4.0" -isystem "$root/lib/boost_1.87.0" \
  -isystem "$root/lib/sundials_6.1.1/include" \
  -isystem "$root/lib/sundials_6.1.1/src/sundials" \
  "$here/bench.cpp" -o "$here/bench" \
  "$root/lib/sundials_6.1.1/lib/libsundials_cvodes.a" \
  "$root/lib/sundials_6.1.1/lib/libsundials_idas.a" \
  "$root/lib/sundials_6.1.1/lib/libsundials_nvecserial.a" \
  -L "$root/lib/tbb" -ltbb -Wl,-rpath,"$root/lib/tbb"
