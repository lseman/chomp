#pragma once

#include <string>

#include <blocks/filter.h>

// -----------------------------------------------------------------------------
// Trust-region configuration
// -----------------------------------------------------------------------------
struct TRConfig {
    std::string tr_method = "auto";  // "auto" | "pcg" | "dogleg" | "gltr"
    int dogleg_n_max = 300;          // use dogleg if n <= this and dense SPD
    int gltr_n_min = 200;            // use GLTR if n >= this
    double gltr_cond_trigger = 1e8;  // or when cg stalls / high cond estimate
    bool assume_gn_spd = true;       // okay to treat GN H as SPD for dogleg

    double delta0 = 1.0, delta_min = 1e-10, delta_max = 1e3;
    double cg_tol = 1e-8, cg_tol_rel = 1e-4;
    int cg_maxiter = 200;
    double neg_curv_tol = 1e-14, rcond = 1e-12, metric_shift = 1e-10;
    // ARC-style adaptive regularization
    double arc_reg_min = 1e-10;
    double arc_reg_max = 1e4;
    double arc_reg_update_factor = 2.0;
    double arc_curvature_tol = -1e-6;
    double zeta = 0.8, constraint_tol = 1e-8;
    int max_active_set_iter = 8;
    bool use_prec = true;
    std::string prec_kind = "ssor";
    std::string norm_type = "euclid";
    double ssor_omega = 1.0;
    double gamma1 = 0.5, gamma2 = 2.0, eta1 = 0.1, eta2 = 0.9;
    bool curvature_aware = true;

    // criticality
    bool criticality_enabled = true;
    int max_crit_shrinks = 1;
    double kappa_g = 1e-2;
    double theta_crit = 0.5;
    double kkt_residual_tol = 1e-6, complementarity_tol = 1e-8,
           licq_threshold = 1e-10;
    bool use_kkt_criticality = true, use_curvature_criticality = true;
    double curvature_criticality_tol = 1e-6;

    // box
    std::string box_mode = "alpha";  // "alpha" | "projection"
    bool recover_lam_active_only = true;

    // filter
    bool use_filter = true;
    FilterConfig filter_cfg{};

    // SOC / funnel
    bool use_soc = true;
    double soc_theta_reduction = 0.9;
    double soc_radius_fraction = 0.5;
    bool soc_use_funnel = true;
    double funnel_gamma = 0.01;

    // extras
    double jacobian_reg_min = 1e-12;
    double constraint_weight = 0.3;
    bool feasibility_restoration = true;
    int restoration_max_iter = 4;
    double restoration_eta = 1e-4;
    double restoration_backtrack = 0.5;
    double restoration_weight_eq = 10.0;
    double restoration_weight_ineq = 1.0;
    bool adaptive_eta_thresholds = true;
    double eta_adaptive_factor = 0.1;
    bool constraint_based_growth = false;
    double max_jacobian_condition = 1e10;
};
