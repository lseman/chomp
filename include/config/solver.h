#pragma once

#include "core/types.h"

#include <string>

struct SolverInfo {
    std::string mode{"ip"};
    double step_norm{0.0};
    bool accepted{true};
    bool converged{true};

    double f{0.0};
    double theta{0.0};
    double stat{0.0};
    double ineq{0.0};
    double eq{0.0};
    double comp{0.0};

    int ls_iters{0};
    double alpha{0.0};
    double rho{0.0};
    double tr_radius{0.0};
    double delta{0.0};
    double mu{0.0};


    double penalty_rho{0.0};
    double step_quality_ratio{0.0};
    bool was_clipped{false};
    int clip_streak{0};
    int good_streak{0};
    double theta_reduction{0.0};
};
