#pragma once

#include "core/types.h"

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>

class ModelC;

struct KKT {
    double stat{0.0}, eq{0.0}, ineq{0.0}, comp{0.0};
};

namespace kkt {

struct KKTConfig {
    double cg_tol = 1e-4;
    int cg_maxit = 100;
    double ip_hess_reg0 = 1e-8;
    double schur_dense_cutoff = 0.25;
    std::string prec_type = "ssor";
    double ssor_omega = 1.0;
    std::string sym_ordering = "amd_custom";
    bool use_simd = true;
    int block_size = 256;
    bool adaptive_gamma = false;
    bool assemble_schur_if_m_small = true;
    bool use_prec = true;
    double delta_min = 1e-12;
    double delta_max = 1e+6;

    double schur_delta2_min = 1e-12;
    double schur_delta2_max = 1e-2;
    int cg_maxit2 = 2 * 200;

    double amd_dense_cutoff = 0.1;
    bool amd_dense_cutoff_has_value = false;
    bool use_hvp = false;
    std::optional<double> hvp_smw_threshold = 1e-3;
    std::optional<double> hvp_iterative_tol = 1e-10;
    std::optional<int> hvp_iterative_maxiter = std::nullopt;
};

struct KKTReusable {
    virtual ~KKTReusable() = default;
    virtual std::pair<dvec, dvec> solve(const dvec& r1,
                                        const std::optional<dvec>& r2,
                                        double cg_tol = 1e-8,
                                        int cg_maxit = 200) = 0;
};

struct KKTStrategy {
    virtual ~KKTStrategy() = default;
    virtual std::tuple<dvec, dvec, std::shared_ptr<KKTReusable>>
    factor_and_solve(ModelC* model_in, const spmat& W,
                     const std::optional<spmat>& G, const dvec& r1,
                     const std::optional<dvec>& r2, const KKTConfig& cfg,
                     std::optional<double> regularizer,
                     std::unordered_map<std::string, dvec>& cache,
                     double delta = 0.0,
                     std::optional<double> gamma = std::nullopt,
                     bool assemble_schur_if_m_small = true,
                     bool use_prec = true) = 0;
    std::string name;
};

}  // namespace kkt
