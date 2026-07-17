// ip_cpp.cpp — refactored and organized C++23 helpers
#pragma once
#include <core/definitions.h>
#include <model/model.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SparseCore>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef IP_PROFILE
#define IP_PROFILE 0
#endif

#if IP_PROFILE
struct ScopedTimer {
    using clock = std::chrono::high_resolution_clock;
    std::string name;
    clock::time_point t0, last;
    bool print_on_destroy;

    explicit ScopedTimer(std::string n, bool print_total = true)
        : name(std::move(n)),
          t0(clock::now()),
          last(t0),
          print_on_destroy(print_total) {
        std::cout << "[IP] >>> " << name << " begin\n";
    }
    ~ScopedTimer() {
        if (print_on_destroy) {
            auto dt =
                std::chrono::duration<double, std::milli>(clock::now() - t0)
                    .count();
            std::cout << "[IP] <<< " << name << " end  (total: " << std::fixed
                      << std::setprecision(3) << dt << " ms)\n";
        }
    }
    void lap(const char* label) {
        auto now = clock::now();
        auto dt = std::chrono::duration<double, std::milli>(now - last).count();
        std::cout << "      [lap] " << label << ": " << std::fixed
                  << std::setprecision(3) << dt << " ms\n";
        last = now;
    }
};
#define IP_TIMER(name) ScopedTimer __ip_timer__(name)
#define IP_LAP(label) __ip_timer__.lap(label)
#define IP_LOG(msg) std::cout << "[IP] " << msg << "\n"
#else
struct ScopedTimerNoop {
    explicit ScopedTimerNoop(const char*) {}
    void lap(const char*) {}
};
#define IP_TIMER(name) ScopedTimerNoop __ip_timer__(name)
#define IP_LAP(label) \
    do {              \
    } while (0)
#define IP_LOG(msg) \
    do {            \
    } while (0)
#endif

namespace nb = nanobind;

// ---------- Constants ----------
namespace consts {
constexpr double EPS_DIV = 1e-16;
constexpr double EPS_POS = 1e-12;
constexpr double INF = std::numeric_limits<double>::infinity();
}  // namespace consts

// ---------- External structs ----------
struct StepResult {
    dvec x, lam, nu;
};
struct KKTResult {
    dvec dx, dy;
    std::shared_ptr<kkt::KKTReusable> reusable;
};

// ---------- Small utilities ----------
template <class T>
[[nodiscard]] constexpr T clamp(T v, T lo, T hi) noexcept {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}
template <class T>
[[nodiscard]] constexpr T clamp_min(T v, T lo) noexcept {
    return (v < lo) ? lo : v;
}
template <class T>
[[nodiscard]] constexpr T clamp_max(T v, T hi) noexcept {
    return (v > hi) ? hi : v;
}
template <class T>
[[nodiscard]] constexpr T clamp01(T v) noexcept {
    return v < T(0) ? T(0) : (v > T(1) ? T(1) : v);
}

[[nodiscard]] inline double sdiv(double num, double den,
                                 double eps = consts::EPS_DIV) noexcept {
    const double d = (std::abs(den) < eps) ? (den < 0 ? -eps : eps) : den;
    return num / d;
}

[[nodiscard]] inline double safe_inf_norm(const dvec& v) noexcept {
    return (v.size() == 0) ? 0.0 : v.cwiseAbs().maxCoeff();
}

static inline double percentile_(const dvec& v, double p01_99) {
    if (v.size() == 0) return 0.0;
    std::vector<double> a(v.data(), v.data() + v.size());
    std::sort(a.begin(), a.end());
    p01_99 = clamp(p01_99, 0.0, 1.0);
    const double idx = p01_99 * (a.size() - 1);
    const size_t i = static_cast<size_t>(idx);
    const double frac = idx - i;
    if (i + 1 < a.size()) return a[i] * (1.0 - frac) + a[i + 1] * frac;
    return a.back();
}

// ---------- Core data structures ----------
struct IPState {
    int mI = 0, mE = 0;
    dvec s, lam, nu, zL, zU;
    double mu = 1e-2;
    bool initialized = false;
};

struct Bounds {
    dvec lb, ub, sL, sU;
    std::vector<uint8_t> hasL, hasU;
};

struct Sigmas {
    dvec Sigma_x, Sigma_s;
};

struct EvalPack {
    double f{};
    dvec g, cI, cE;
    spmat JI, JE;
    double theta{};
};

#include "soc.h"
#include "mg_solver.h"
