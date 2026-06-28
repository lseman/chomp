// ---------- SOC helpers ----------
struct SOCResult {
    dvec   q;
    bool   applied = false;
    double theta0  = NAN;
    double theta1  = NAN;
};

[[nodiscard]] dvec clip_correction_to_radius_(const dvec &s, const dvec &q) const {
    if (tr_norm_(s + q) <= delta_ + 1e-14 || tr_norm_(q) <= 1e-16) return q;
    const double t = detail::boundary_intersection_metric(metric_, s, q, delta_);
    return std::clamp(t, 0.0, 1.0) * q;
}

[[nodiscard]] SOCResult soc_correction_(ModelC *model, const dvec &x, const dvec &p, const LinOp &Hop,
                                        const std::optional<dmat> &Hdense, const std::optional<spmat> &Hsparse,
                                        const dvec &lam_ineq, double mu, double wE, double wI, double tolE,
                                        double violI, double reg, double sigma0, const std::optional<dvec> &lb,
                                        const std::optional<dvec> &ub) {
    (void)Hop;
    (void)lam_ineq;
    (void)mu;
    (void)lb;
    (void)ub;

    SOCResult R{dvec::Zero(p.size()), false, NAN, NAN};

    const dvec               x_trial = x + p;
    std::vector<std::string> need{"cE", "cI", "JE", "JI"};
    model->eval_all(x_trial, /*need*/ need);

    const double theta0 = model->constraint_violation(x_trial);
    R.theta0            = theta0;
    if (theta0 <= cfg_.constraint_tol) return R;

    const int mI = model->get_mI();
    const int mE = model->get_mE();
    const int n  = model->get_n();

    const dmat JI_ = (mI > 0) ? model->get_JI().value() : dmat(mI, n);
    const dmat JE_ = (mE > 0) ? model->get_JE().value() : dmat(mE, n);
    const dvec cI_ = (mI > 0) ? model->get_cI().value() : dvec::Zero(mI);
    const dvec cE_ = (mE > 0) ? model->get_cE().value() : dvec::Zero(mE);

    jacobian_condition_ = estimate_jacobian_condition_(JE_, JI_, cI_, cfg_.constraint_tol);

    dvec q = compute_soc_step_(cE_, cI_, JE_, JI_, wE, wI, tolE, violI, std::max(reg, cfg_.jacobian_reg_min), sigma0,
                               Hdense, Hsparse);

    // Bound within SOC radius and TR metric
    const double soc_radius = cfg_.soc_radius_fraction * delta_;
    const double qn         = tr_norm_(q);
    if (qn > soc_radius && qn > 1e-16) q *= (soc_radius / qn);

    q = clip_correction_to_radius_(p, q);
    if (box_.active()) q = box_.enforce_correction(x_trial, q);

    const double theta1 = model->constraint_violation(x_trial + q);
    R.theta1            = theta1;

    if (soc_acceptance_test_(theta0, theta1)) {
        R.q       = q;
        R.applied = true;
        if (cfg_.soc_use_funnel) current_theta_ = theta1;
    }
    return R;
}

[[nodiscard]] double adjust_growth_for_curvature_(double growth, double curv) const {
    if (!std::isfinite(curv)) return growth;
    if (curv <= -1e-10) return 1.0;
    if (curv < 1e-12) return std::min(1.2, growth);
    if (curv < 1e-4) return std::min(2.0, std::max(1.2, growth));
    if (curv < 1e-2) return std::min(2.5, std::max(1.4, growth));
    return std::min(3.0, std::max(1.6, 1.25 * growth));
}

// ---------- Regularized LS (dense SPD small systems) ----------
// [[nodiscard]] dvec solve_regularized_ls_(const dmat &A, const dvec &b,
//                                          double reg) const {
//     if (A.size() == 0)
//         return dvec::Zero(b.size());
//     dmat N = A.transpose() * A;
//     N.diagonal().array() += reg;
//     return N.ldlt().solve(A.transpose() * b);
// }

[[nodiscard]] dvec solve_regularized_ls_(const dmat &A, const dvec &b, double reg) const {
    if (A.size() == 0) return dvec::Zero(b.size());

    // Build normal equations with a tiny floor on reg for numerical safety
    const double reg_eff = std::max(reg, 1e-12);
    dmat         N       = A.transpose() * A;
    N.diagonal().array() += reg_eff;

    const dvec rhs = A.transpose() * b;

    // Use the QDLDL-based symmetric SPD solver on the upper triangle
    return solve_sym_spd_qdldl(N, rhs);
}

[[nodiscard]] dvec solve_regularized_metric_ls_(const dmat &A, const dvec &b, const dmat &M, double reg) const {
    if (A.size() == 0) return dvec::Zero(M.cols()); // n=cols(M). (No rows in A ⇒ x=0)

    try {
        // Factor M = L Lᵀ (with minimal shift inside helper for robustness)
        const double reg_eff = std::max(reg, 1e-12);
        const dmat   L       = psd_cholesky_with_shift(M, reg_eff); // Lower-triangular

        // We need: (A M^{-1} Aᵀ + reg I) y = b, then x = M^{-1} Aᵀ y
        // Compute Minv Aᵀ via two triangular solves on a dense multi-RHS:
        //   Solve L * Y = Aᵀ          → Y = L^{-1} Aᵀ
        //   Solve Lᵀ * Z = Y          → Z = L^{-T} Y = M^{-1} Aᵀ
        const dmat At = A.transpose();
        dmat       Y  = L.triangularView<Eigen::Lower>().solve(At);
        dmat       Z  = L.transpose().triangularView<Eigen::Upper>().solve(Y); // Z = M^{-1}Aᵀ

        // Build S = A * Z = A * M^{-1} * Aᵀ, add Tikhonov reg on the
        // diagonal
        dmat S = A * Z;
        S.diagonal().array() += reg_eff;

        // Solve S y = b with QDLDL (upper triangle packed); fallback inside
        // helper
        const dvec y = solve_sym_spd_qdldl(S, b);

        // Recover x = M^{-1} Aᵀ y efficiently:
        //   t = Aᵀ y
        //   Solve L * w = t, then Lᵀ * x = w  → x = M^{-1} Aᵀ y
        const dvec t = At * y;
        dvec       w = L.triangularView<Eigen::Lower>().solve(t);
        dvec       x = L.transpose().triangularView<Eigen::Upper>().solve(w);
        return x;
    } catch (...) {
        // If anything goes wrong with metric factorization, fall back to
        // Euclidean LS
        return solve_regularized_ls_(A, b, reg);
    }
}

// ---------- Jacobian condition estimate (stacked blocks, tight) ----------
[[nodiscard]] double estimate_jacobian_condition_(const std::optional<dmat> &JE, const std::optional<dmat> &JI,
                                                  const std::optional<dvec> &cI, double tol) const {
    std::vector<dmat> blocks;
    blocks.reserve(2);

    if (JE && JE->size()) blocks.push_back(*JE);

    if (JI && JI->size() && cI) {
        const auto mask = (cI->array().abs() <= 10 * tol);
        if (mask.any()) {
            const int m = (int)mask.count();
            dmat      JI_act(m, JI->cols());
            for (int i = 0, k = 0; i < cI->size(); ++i)
                if (mask[i]) JI_act.row(k++) = JI->row(i);
            blocks.push_back(std::move(JI_act));
        }
    }

    if (blocks.empty()) return 1.0;

    int rows = 0, cols = blocks[0].cols();
    for (const auto &B : blocks) rows += B.rows();

    dmat J(rows, cols);
    for (int ofs = 0, i = 0; i < (int)blocks.size(); ++i) {
        J.middleRows(ofs, blocks[i].rows()) = blocks[i];
        ofs += blocks[i].rows();
    }

    try {
        Eigen::JacobiSVD<dmat> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const auto            &s = svd.singularValues();
        if (s.size() == 0 || s.tail(1)[0] <= 1e-16) return cfg_.max_jacobian_condition;
        return std::min(s.head(1)[0] / s.tail(1)[0], cfg_.max_jacobian_condition);
    } catch (...) { return cfg_.max_jacobian_condition; }
}

// ---------- SOC acceptance ----------
[[nodiscard]] bool soc_acceptance_test_(double theta0, double theta1) const {
    if (!std::isfinite(theta0) || !std::isfinite(theta1)) return false;
    if (cfg_.soc_use_funnel) {
        const double sufficient = cfg_.soc_theta_reduction * theta0;
        const double funnel_ref = std::isfinite(current_theta_) ? current_theta_ : theta0;
        const double funnel     = cfg_.funnel_gamma * funnel_ref;
        return theta1 <= std::max(sufficient, funnel);
    }
    return theta1 <= cfg_.soc_theta_reduction * theta0;
}

// ---------- SOC step (single pass build, minimal temporaries) ----------
[[nodiscard]] dvec compute_soc_step_(const std::optional<dvec> &cE, const std::optional<dvec> &cI,
                                     const std::optional<dmat> &JE, const std::optional<dmat> &JI, double wE, double wI,
                                     double tolE, double violI, double reg, double sigma0,
                                     const std::optional<dmat> &Hdense, const std::optional<spmat> &Hsparse) const {
    std::vector<dmat> Jblocks;
    Jblocks.reserve(2);
    std::vector<dvec> rblocks;
    rblocks.reserve(2);

    const double sE = std::sqrt(std::max(0.0, wE));
    const double sI = std::sqrt(std::max(0.0, wI));

    if (cE && JE && JE->size()) {
        const auto mask = (cE->array().abs() > tolE);
        if (mask.any()) {
            const int m = (int)mask.count();
            dmat      J(m, JE->cols());
            dvec      r(m);
            for (int i = 0, k = 0; i < cE->size(); ++i)
                if (mask[i]) {
                    J.row(k) = sE * JE->row(i);
                    r[k++]   = -sE * (*cE)[i];
                }
            Jblocks.push_back(std::move(J));
            rblocks.push_back(std::move(r));
        }
    }

    if (cI && JI && JI->size()) {
        const auto mask = (cI->array() > violI);
        if (mask.any()) {
            const int m = (int)mask.count();
            dmat      J(m, JI->cols());
            dvec      r(m);
            for (int i = 0, k = 0; i < cI->size(); ++i)
                if (mask[i]) {
                    J.row(k) = sI * JI->row(i);
                    r[k++]   = -sI * (*cI)[i];
                }
            Jblocks.push_back(std::move(J));
            rblocks.push_back(std::move(r));
        }
    }

    if (Jblocks.empty()) {
        const int n = Hdense ? Hdense->cols() : (Hsparse ? (int)Hsparse->cols() : 0);
        return dvec::Zero(n);
    }

    int rows = 0, cols = Jblocks[0].cols();
    for (const auto &J : Jblocks) rows += J.rows();

    dmat J(rows, cols);
    dvec r(rows);
    for (int ofs = 0, i = 0; i < (int)Jblocks.size(); ++i) {
        J.middleRows(ofs, Jblocks[i].rows()) = Jblocks[i];
        r.segment(ofs, Jblocks[i].rows())    = rblocks[i];
        ofs += Jblocks[i].rows();
    }

    if (cfg_.norm_type == "ellip" && (Hdense || Hsparse)) {
        const dmat H = Hdense ? *Hdense : dmat(*Hsparse);
        const dmat M = symmetrize(H) + sigma0 * dmat::Identity(cols, cols);
        return solve_regularized_metric_ls_(J, r, M, reg * std::max(1.0, jacobian_condition_));
    }
    return solve_regularized_ls_(J, r, reg * std::max(1.0, jacobian_condition_));
}
