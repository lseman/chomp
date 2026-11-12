#pragma once

#include <cmath>
#include <config/solver.h>
#include <fmt/color.h>
#include <fmt/core.h>
#include <string>

namespace nlp {
inline void print_iteration_row(int k, const SolverInfo &info, int last_header_ref, bool force_header = false) {
    using fmt::color;
    using fmt::emphasis;
    using fmt::fg;

    static bool banner_printed = false;

    if (!banner_printed) {
        fmt::print(fg(color::cyan) | emphasis::bold, "\nCHOMP");
        fmt::print(fg(color::light_gray) | emphasis::bold, " — made by ");
        fmt::print(fg(color::white) | emphasis::bold, "L. O. Seman\n");
        banner_printed = true;
    }

    if (force_header || k == 0 || (k - last_header_ref) >= 20) {
        fmt::print(fg(color::light_gray) | emphasis::bold, " {:>3s} {:>3s} {:>12s} {:>13s} {:>11s} {:>9s} {:>9s}\n",
                   "k", "st", "step", "f", "theta", "alpha", "Δ");
    }

    const auto st_col = info.accepted ? color::green : color::red;

    fmt::print(" {:>3d} ", k);
    fmt::print(fg(st_col) | emphasis::bold, "{:>3s} ", info.accepted ? "A" : "R");

    fmt::print("{:>12.3e} ", info.step_norm);
    fmt::print(fg(color::white), "{:>13.6e} ", info.f);
    fmt::print(fg(color::white), "{:>11.3e} ", info.theta);
    fmt::print("{:>9.2e} {:>9.2e}\n", info.alpha, info.tr_radius);
}

} // namespace nlp
