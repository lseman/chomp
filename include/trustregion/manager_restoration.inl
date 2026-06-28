    [[nodiscard]] AcceptanceTrial restoration_phase_(
        ModelC* model, const std::optional<dvec>& x,
        const std::optional<dvec>& lb, const std::optional<dvec>& ub,
        double delta, const std::optional<double>& f_old,
        const std::optional<double>& theta_old) {
        if (!cfg_.feasibility_restoration || !model || !x || !theta_old ||
            !std::isfinite(*theta_old) || *theta_old <= cfg_.constraint_tol) {
            return AcceptanceTrial{.accepted = false,
                                   .step = x ? dvec::Zero(x->size()) : dvec()};
        }

        const std::vector<std::string> need{"f", "cE", "cI", "JE", "JI"};
        model->eval_all(*x, need);

        const int mI = model->get_mI();
        const int mE = model->get_mE();
        const int n = model->get_n();

        const auto f_base = f_old ? f_old : model->get_f();
        const std::optional<dvec> cE =
            (mE > 0) ? std::optional<dvec>(model->get_cE().value())
                     : std::nullopt;
        const std::optional<dvec> cI =
            (mI > 0) ? std::optional<dvec>(model->get_cI().value())
                     : std::nullopt;
        const std::optional<dmat> JE =
            (mE > 0) ? std::optional<dmat>(dmat(model->get_JE().value()))
                     : std::nullopt;
        const std::optional<dmat> JI =
            (mI > 0) ? std::optional<dmat>(dmat(model->get_JI().value()))
                     : std::nullopt;

        dvec d_rest = compute_soc_step_(
            cE, cI, JE, JI, cfg_.restoration_weight_eq,
            cfg_.restoration_weight_ineq, cfg_.constraint_tol, 0.0,
            std::max(cfg_.jacobian_reg_min, 1e-12), 0.0, std::nullopt,
            std::nullopt);
        if (d_rest.size() != n || !d_rest.allFinite())
            return AcceptanceTrial{.accepted = false,
                                   .step = dvec::Zero(n)};

        const double dn = tr_norm_(d_rest);
        if (dn > delta && dn > 1e-16) d_rest *= (delta / dn);
        if (box_.active()) d_rest = box_.enforce_step(d_rest);
        if (tr_norm_(d_rest) <= 1e-14)
            return AcceptanceTrial{.accepted = false,
                                   .step = dvec::Zero(n)};

        const auto merit0 = exact_penalty_merit_(f_base, theta_old);
        double alpha = 1.0;
        const int tries = std::max(1, cfg_.restoration_max_iter);
        for (int k = 0; k < tries; ++k) {
            dvec sa = alpha * d_rest;
            if (box_.active()) sa = box_.enforce_step(sa);
            if (tr_norm_(sa) <= 1e-14) break;

            const double theta_pred = linearized_constraint_violation_(
                cE, cI, JE, JI, sa);
            auto [f_t, th_t] = eval_model_f_theta_(model, x, sa);
            if (!(th_t && std::isfinite(*th_t))) {
                alpha *= cfg_.restoration_backtrack;
                continue;
            }

            const bool filter_ok = filter_accepts_(f_t, th_t, delta);
            const double theta_floor =
                *theta_old - cfg_.restoration_eta * alpha *
                                 std::max(1.0, *theta_old);
            const bool theta_improves = *th_t <= theta_floor;
            const auto merit_t = exact_penalty_merit_(f_t, th_t);
            const bool merit_improves =
                merit0 && merit_t &&
                (*merit_t <=
                 *merit0 -
                     1e-4 * alpha * std::max(1.0, std::abs(*merit0)));

            if (filter_ok || theta_improves || merit_improves) {
                if (filter_ok)
                    (void)filter_.add_if_acceptable(*th_t, *f_t, delta);
                return AcceptanceTrial{
                    .accepted = true,
                    .step = std::move(sa),
                    .accepted_by =
                        filter_ok ? "restoration-filter" : "restoration",
                    .f_trial = f_t,
                    .theta_trial = th_t,
                    .predicted =
                        cfg_.constraint_weight *
                        std::max(0.0, *theta_old - theta_pred),
                    .actual = cfg_.constraint_weight *
                              std::max(0.0, *theta_old - *th_t)};
            }

            alpha *= cfg_.restoration_backtrack;
        }

        return AcceptanceTrial{.accepted = false,
                               .step = dvec::Zero(n),
                               .accepted_by = "rejected"};
    }
