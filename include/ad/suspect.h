#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ad_bindings.h"
#include "ad_graph.h"
#include "definitions.h"

#define NDEBUG

/* ============================================================================
   SUSPECT-style Convexity/Monotonicity Analyzer (enhanced)
   - Interval forward/backward propagation
   - Monotonicity & convexity tags + simple rules
   - Plugin detectors: SOC norm, quadratic (x^2, sum of squares), fractional-affine
   - Graph-agnostic via SuspectTraits<GraphT,NodeT>
   - Clean ASCII summary printer
   ============================================================================ */

namespace suspect {

// ----------------------------- Tags -----------------------------------------
enum class CvxTag { Convex, Concave, Affine, Unknown };
enum class MonoTag { Nondecreasing, Nonincreasing, Unknown };

inline const char *to_string(CvxTag c) {
    switch (c) {
    case CvxTag::Convex: return "convex";
    case CvxTag::Concave: return "concave";
    case CvxTag::Affine: return "affine";
    default: return "unknown";
    }
}
inline const char *to_string(MonoTag m) {
    switch (m) {
    case MonoTag::Nondecreasing: return "nondecreasing";
    case MonoTag::Nonincreasing: return "nonincreasing";
    default: return "unknown";
    }
}

// --------------------------- Interval ---------------------------------------
struct Interval {
    double lo{-std::numeric_limits<double>::infinity()};
    double hi{std::numeric_limits<double>::infinity()};

    Interval() = default;
    Interval(double L, double H) : lo(L), hi(H) {
        if (lo > hi) { lo = hi = std::numeric_limits<double>::quiet_NaN(); }
    }

    static Interval Full() { return {}; }
    static Interval Empty() {
        Interval I;
        I.lo = I.hi = std::numeric_limits<double>::quiet_NaN();
        return I;
    }
    static Interval Point(double v) { return Interval(v, v); }

    bool   empty() const { return std::isnan(lo) || std::isnan(hi); }
    bool   singleton() const { return !empty() && std::abs(hi - lo) <= 1e-12 * std::max(1.0, std::abs(lo)); }
    double width() const { return empty() ? 0.0 : (hi - lo); }
    bool   containsZero() const { return !empty() && lo <= 0.0 && 0.0 <= hi; }
    bool   positive() const { return !empty() && lo > 0.0; }
    bool   nonnegative() const { return !empty() && lo >= 0.0; }
    bool   negative() const { return !empty() && hi < 0.0; }
    bool   nonpositive() const { return !empty() && hi <= 0.0; }

    Interval intersect(const Interval &o) const {
        if (empty() || o.empty()) return Empty();
        double L = std::max(lo, o.lo), H = std::min(hi, o.hi);
        return (L <= H) ? Interval(L, H) : Empty();
    }
    Interval hull(const Interval &o) const {
        if (empty()) return o;
        if (o.empty()) return *this;
        return Interval(std::min(lo, o.lo), std::max(hi, o.hi));
    }
    Interval add(const Interval &o) const {
        if (empty() || o.empty()) return Empty();
        return Interval(lo + o.lo, hi + o.hi);
    }
    Interval neg() const {
        if (empty()) return Empty();
        return Interval(-hi, -lo);
    }
    Interval mul(const Interval &o) const {
        if (empty() || o.empty()) return Empty();
        double a = lo * o.lo, b = lo * o.hi, c = hi * o.lo, d = hi * o.hi;
        double L = std::min(std::min(a, b), std::min(c, d));
        double H = std::max(std::max(a, b), std::max(c, d));
        return Interval(L, H);
    }
    Interval reciprocal() const {
        if (empty()) return Empty();
        // If [lo,hi] crosses 0, we conservatively return Full.
        if (containsZero()) return Full();
        return Interval(1.0 / hi, 1.0 / lo);
    }
    Interval div(const Interval &o) const {
        if (empty() || o.empty()) return Empty();
        if (o.containsZero()) return Full();
        return mul(o.reciprocal());
    }
    Interval sqrt() const {
        if (empty() || hi < 0.0) return Empty();
        return Interval(std::sqrt(std::max(0.0, lo)), std::sqrt(std::max(0.0, hi)));
    }
    Interval exp() const {
        if (empty()) return Empty();
        return Interval(std::exp(lo), std::exp(hi));
    }
    Interval log() const {
        if (empty() || hi <= 0.0) return Empty();
        return Interval(std::log(std::max(lo, 1e-300)), std::log(hi));
    }
    Interval pow(double p) const {
        if (empty()) return Empty();
        if (std::abs(p) < 1e-12) return Point(1.0); // x^0

        std::vector<double> cand;
        auto push = [&](double v) { if (std::isfinite(v)) cand.push_back(v); };
        auto isInt  = [](double t) { return std::abs(t - std::round(t)) < 1e-12; };
        auto tryPow = [&](double x) {
            if (x > 0.0 || (x == 0.0 && p > 0.0) || isInt(p)) {
                push(std::pow(std::abs(x), p) * (x < 0.0 && static_cast<int>(std::round(p)) % 2 == 1 ? -1.0 : 1.0));
            }
        };

        tryPow(lo);
        tryPow(hi);
        if (lo <= 0.0 && hi >= 0.0) {
            if (p > 0.0 && p < 1.0) {
                push(0.0);
            } else if (isInt(p) && (static_cast<long long>(std::llround(p)) % 2 == 0)) {
                push(0.0);
            } else if (lo < 0.0 && !isInt(p)) {
                push(0.0);
            }
        }

        if (cand.empty()) return Full();
        auto mm = std::minmax_element(cand.begin(), cand.end());
        return Interval(*mm.first, *mm.second);
    }

    std::string to_string() const {
        if (empty()) return "[empty]";
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6) << "[" << lo << ", " << hi << "]";
        return oss.str();
    }
};

// ---------------------------- NodeInfo --------------------------------------
struct NodeInfo {
    Interval    bounds{Interval::Full()};
    MonoTag     mono{MonoTag::Unknown};
    CvxTag      cvx{CvxTag::Unknown};
    bool        is_linear{false};
    bool        is_quadratic{false};
    int         poly_degree{-1};
    std::string hint;
};

// ------------------------ Plugin Detector Hook ------------------------------
template <class GraphT, class NodeT>
struct ISuspectDetector {
    virtual ~ISuspectDetector() = default;
    virtual bool detect(const GraphT &G, const NodeT &n, const std::vector<NodeInfo> &childInfos, NodeInfo &info) = 0;
};

// ---------------------- SuspectTraits forward decl --------------------------
template <class GraphT, class NodeT>
struct SuspectTraits;

template <class Traits, class NodeT>
static bool is_affine_expr_(const NodeT& n) {
    auto op = Traits::op(n);
    if (Traits::is_const(n) || Traits::is_var(n)) return true;

    auto ins = Traits::inputs(n);
    switch (op) {
    case Traits::Op::Neg:
        return ins.size() == 1 && is_affine_expr_<Traits>(ins[0]);
    case Traits::Op::Add:
        if (ins.empty()) return true;
        for (auto& child : ins) {
            if (!is_affine_expr_<Traits>(child)) return false;
        }
        return true;
    case Traits::Op::Subtract:
        return ins.size() == 2 && is_affine_expr_<Traits>(ins[0]) &&
               is_affine_expr_<Traits>(ins[1]);
    case Traits::Op::Multiply: {
        int nonconst = 0;
        std::optional<NodeT> affine_child;
        for (auto& child : ins) {
            if (Traits::is_const(child)) continue;
            nonconst++;
            if (!affine_child.has_value()) affine_child = child;
        }
        return nonconst <= 1 &&
               (!affine_child.has_value() || is_affine_expr_<Traits>(*affine_child));
    }
    case Traits::Op::Divide:
        return ins.size() == 2 && is_affine_expr_<Traits>(ins[0]) &&
               Traits::is_const(ins[1]) &&
               std::abs(Traits::const_value(ins[1])) > 1e-12;
    default:
        return false;
    }
}

template <class Traits, class NodeT>
static bool is_affine_square_expr_(const NodeT& n) {
    auto op = Traits::op(n);
    if (op == Traits::Op::Multiply) {
        auto ti = Traits::inputs(n);
        return ti.size() == 2 && Traits::id(ti[0]) == Traits::id(ti[1]) &&
               is_affine_expr_<Traits>(ti[0]);
    }
    if (op == Traits::Op::Pow) {
        auto ti = Traits::inputs(n);
        return std::abs(Traits::pow_exponent(n) - 2.0) < 1e-12 &&
               !ti.empty() && is_affine_expr_<Traits>(ti[0]);
    }
    return false;
}

template <class Traits, class NodeT>
static bool is_sum_affine_squares_plus_nonneg_constant_(const NodeT& n,
                                                        bool require_constant_term) {
    if (Traits::op(n) != Traits::Op::Add) return false;
    bool saw_constant = !require_constant_term;
    bool saw_square = false;
    for (auto& t : Traits::inputs(n)) {
        if (Traits::is_const(t)) {
            if (Traits::const_value(t) < 0.0) return false;
            saw_constant = true;
            continue;
        }
        if (!is_affine_square_expr_<Traits>(t)) return false;
        saw_square = true;
    }
    return saw_constant && saw_square;
}

template <class Traits, class NodeT>
static std::optional<std::pair<NodeT, double>> extract_power_rewrite_(const NodeT& n) {
    if (Traits::op(n) != Traits::Op::Exp) return std::nullopt;
    auto exp_inputs = Traits::inputs(n);
    if (exp_inputs.size() != 1) return std::nullopt;

    const NodeT& mul = exp_inputs[0];
    if (Traits::op(mul) != Traits::Op::Multiply) return std::nullopt;

    double scale = 1.0;
    std::optional<NodeT> log_node;
    for (auto& child : Traits::inputs(mul)) {
        if (Traits::is_const(child)) {
            scale *= Traits::const_value(child);
            continue;
        }
        if (Traits::op(child) == Traits::Op::Log && !log_node.has_value()) {
            log_node = child;
            continue;
        }
        return std::nullopt;
    }

    if (!log_node.has_value()) return std::nullopt;
    auto log_inputs = Traits::inputs(*log_node);
    if (log_inputs.size() != 1) return std::nullopt;
    return std::make_pair(log_inputs[0], scale);
}

// ============================== Built-in detectors ==========================
template <class GraphT, class NodeT>
struct SocNormDetector final : ISuspectDetector<GraphT, NodeT> {
    using T = SuspectTraits<GraphT, NodeT>;
    bool detect(const GraphT&, const NodeT& n,
                const std::vector<NodeInfo>&, NodeInfo& info) override {
        auto op  = T::op(n);
        auto ins = T::inputs(n);

        auto looksLikeSumSquares = [&](const NodeT& s) -> bool {
            return is_sum_affine_squares_plus_nonneg_constant_<T>(s, false);
        };

        // sqrt(sum_i x_i^2)
        if (op == T::Op::Sqrt && !ins.empty() && looksLikeSumSquares(ins[0])) {
            info.cvx = CvxTag::Convex; info.hint = "Euclidean norm is convex.";
            return true;
        }
        // pow(sum_i x_i^2, 0.5)
        if (op == T::Op::Pow && ins.size()>=2 && T::is_const(ins[1])) {
            double p = T::pow_exponent(n);
            if (std::abs(p - 0.5) < 1e-12 && looksLikeSumSquares(ins[0])) {
                info.cvx = CvxTag::Convex; info.hint = "||x||_2 is convex.";
                return true;
            }
            if (std::abs(p - 0.5) < 1e-12 &&
                is_sum_affine_squares_plus_nonneg_constant_<T>(ins[0], true)) {
                info.cvx = CvxTag::Convex;
                info.hint = "Shifted Euclidean norm is convex.";
                return true;
            }
        }
        if (op == T::Op::Sqrt && !ins.empty() &&
            is_sum_affine_squares_plus_nonneg_constant_<T>(ins[0], true)) {
            info.cvx = CvxTag::Convex;
            info.hint = "Shifted Euclidean norm is convex.";
            return true;
        }
        if (auto power = extract_power_rewrite_<T>(n)) {
            const auto& [base, p] = *power;
            if (std::abs(p - 0.5) < 1e-12 &&
                is_sum_affine_squares_plus_nonneg_constant_<T>(base, true)) {
                info.cvx = CvxTag::Convex;
                info.hint = "Shifted Euclidean norm is convex.";
                return true;
            }
            if (std::abs(p - 0.5) < 1e-12 &&
                is_sum_affine_squares_plus_nonneg_constant_<T>(base, false)) {
                info.cvx = CvxTag::Convex;
                info.hint = "Euclidean norm is convex.";
                return true;
            }
        }
        return false;
    }
};

template <class GraphT, class NodeT>
struct QuadraticSimpleDetector final : ISuspectDetector<GraphT, NodeT> {
    using T = SuspectTraits<GraphT, NodeT>;
    bool detect(const GraphT&, const NodeT& n,
                const std::vector<NodeInfo>& kids, NodeInfo& info) override {
        auto op  = T::op(n);
        auto ins = T::inputs(n);

        if (op == T::Op::Multiply && ins.size()==2 && T::id(ins[0])==T::id(ins[1]) && !kids.empty()) {
            if (kids[0].cvx == CvxTag::Affine ||
                (kids[0].bounds.nonnegative() &&
                 (kids[0].cvx == CvxTag::Convex || kids[0].cvx == CvxTag::Affine)) ||
                (kids[0].bounds.nonpositive() &&
                 (kids[0].cvx == CvxTag::Concave || kids[0].cvx == CvxTag::Affine))) {
                info.cvx = CvxTag::Convex; info.hint = "x^2 is convex on affine/sign-compatible input.";
                return true;
            }
        }
        if (op == T::Op::Pow && std::abs(T::pow_exponent(n)-2.0)<1e-12) {
            if (kids.size()>=1 && (
                    kids[0].cvx == CvxTag::Affine ||
                    (kids[0].bounds.nonnegative() &&
                     (kids[0].cvx == CvxTag::Convex || kids[0].cvx == CvxTag::Affine)) ||
                    (kids[0].bounds.nonpositive() &&
                     (kids[0].cvx == CvxTag::Concave || kids[0].cvx == CvxTag::Affine))
                )) {
                info.cvx = CvxTag::Convex; info.hint = "x^2 is convex on affine/sign-compatible input.";
                return true;
            }
        }
        if (op == T::Op::Add) {
            bool allSquares = true;
            for (size_t i = 0; i < ins.size(); ++i) {
                auto &ch = ins[i];
                auto oc = T::op(ch);
                bool isSq = (oc==T::Op::Multiply && T::id(T::inputs(ch)[0])==T::id(T::inputs(ch)[1]))
                         || (oc==T::Op::Pow && std::abs(T::pow_exponent(ch)-2.0)<1e-12);
                const bool convex_square =
                    (i < kids.size() && kids[i].cvx == CvxTag::Convex);
                if (!isSq || !convex_square) { allSquares=false; break; }
            }
            if (allSquares) { info.cvx = CvxTag::Convex; info.hint = "Sum of squares is convex."; return true; }
        }
        return false;
    }
};

template <class GraphT, class NodeT>
struct FractionalAffineDetector final : ISuspectDetector<GraphT, NodeT> {
    using T = SuspectTraits<GraphT, NodeT>;
    bool detect(const GraphT&, const NodeT& n,
                const std::vector<NodeInfo>& kids, NodeInfo& info) override {
        auto op  = T::op(n);
        auto ins = T::inputs(n);
        if (op != T::Op::Divide || ins.size()!=2) return false;

        // Safe DCP-style rule: c / concave(positive) is convex if c>=0,
        // concave if c<0, since inv_pos is convex and decreasing.
        if (T::is_const(ins[0]) && (kids[1].cvx == CvxTag::Concave || kids[1].cvx == CvxTag::Affine)
            && kids[1].bounds.positive()) {
            double c = T::const_value(ins[0]);
            info.cvx = (c>=0.0) ? CvxTag::Convex : CvxTag::Concave;
            info.hint = "c / concave positive denominator.";
            return true;
        }
        return false;
    }
};

template <class GraphT, class NodeT>
struct QuadOverLinDetector final : ISuspectDetector<GraphT, NodeT> {
    using T = SuspectTraits<GraphT, NodeT>;
    bool detect(const GraphT&, const NodeT& n,
                const std::vector<NodeInfo>& kids, NodeInfo& info) override {
        auto op  = T::op(n);
        auto ins = T::inputs(n);
        if (op != T::Op::Divide || ins.size() != 2 || kids.size() != 2)
            return false;

        const bool denom_ok =
            kids[1].cvx == CvxTag::Affine && kids[1].bounds.positive();
        if (!denom_ok) return false;

        const bool numer_ok =
            is_affine_square_expr_<T>(ins[0]) ||
            is_sum_affine_squares_plus_nonneg_constant_<T>(ins[0], false);
        if (!numer_ok) return false;

        info.cvx = CvxTag::Convex;
        info.hint = "Quadratic-over-linear form is convex with positive affine denominator.";
        return true;
    }
};

// ----------------- LogSumExp Detector ---------------------------------------
// log(sum(exp(x_i))) is convex (standard DCP atom)
// Detected as: log(exp(x1) + exp(x2) + ... + exp(xk))
template <class GraphT, class NodeT>
struct LogSumExpDetector final : ISuspectDetector<GraphT, NodeT> {
    using T = SuspectTraits<GraphT, NodeT>;
    bool detect(const GraphT&, const NodeT& n,
                const std::vector<NodeInfo>& kids, NodeInfo& info) override {
        auto op = T::op(n);
        auto ins = T::inputs(n);
        // Must be log(sum_of_exps)
        if (op != T::Op::Log || ins.size() != 1) return false;
        const NodeT& inner = ins[0];
        if (T::op(inner) != T::Op::Add || inner.inputs.size() < 2) return false;

        // Check all addends are exp(affine)
        bool all_exps = true;
        for (const auto& term : inner.inputs) {
            if (T::op(term) != T::Op::Exp) { all_exps = false; break; }
        }
        if (!all_exps) return false;

        info.cvx = CvxTag::Convex;
        info.hint = "Log-sum-exp is convex.";
        return true;
    }
};

// ----------------- Entropy Detector -----------------------------------------
// -sum(x * log(x)) = sum(-x * log(x)) is concave (entropy)
// Detected as: -(x * log(x)) or sum(-x_i * log(x_i))
template <class GraphT, class NodeT>
struct EntropyDetector final : ISuspectDetector<GraphT, NodeT> {
    using T = SuspectTraits<GraphT, NodeT>;
    bool detect(const GraphT&, const NodeT& n,
                const std::vector<NodeInfo>& kids, NodeInfo& info) override {
        auto op = T::op(n);
        auto ins = T::inputs(n);

        // Pattern: -x * log(x)  (single term)
        if (op == T::Op::Multiply && ins.size() == 2) {
            // Check if one child is -1 and other is x*log(x)
            for (size_t i = 0; i < 2; ++i) {
                if (T::is_const(ins[i]) && std::abs(T::const_value(ins[i]) + 1.0) < 1e-12) {
                    const NodeT& prod = ins[1 - i];
                    if (T::op(prod) == T::Op::Multiply && prod.inputs.size() == 2) {
                        const auto& a = prod.inputs[0];
                        const auto& b = prod.inputs[1];
                        if (T::op(b) == T::Op::Log && b.inputs.size() == 1 &&
                            a.get() == b.inputs[0].get()) {
                            info.cvx = CvxTag::Concave;
                            info.hint = "Negative x*log(x) = entropy term is concave.";
                            return true;
                        }
                    }
                }
            }
        }

        // Pattern: sum of -x_i * log(x_i) terms
        if (op == T::Op::Add && ins.size() >= 2) {
            bool all_entropy = true;
            for (const auto& term : ins) {
                if (T::op(term) != T::Op::Multiply || term.inputs.size() < 2) {
                    all_entropy = false; break;
                }
                bool is_neg_xlogx = false;
                for (size_t i = 0; i < 2; ++i) {
                    if (T::is_const(term.inputs[i]) &&
                        std::abs(T::const_value(term.inputs[i]) + 1.0) < 1e-12) {
                        const NodeT& xlogx = term.inputs[1 - i];
                        if (T::op(xlogx) == T::Op::Multiply && xlogx.inputs.size() == 2) {
                            const auto& a = xlogx.inputs[0];
                            const auto& b = xlogx.inputs[1];
                            if (T::op(b) == T::Op::Log && b.inputs.size() == 1 &&
                                a.get() == b.inputs[0].get()) {
                                is_neg_xlogx = true;
                                break;
                            }
                        }
                    }
                }
                if (!is_neg_xlogx) { all_entropy = false; break; }
            }
            if (all_entropy) {
                info.cvx = CvxTag::Concave;
                info.hint = "Sum of negative entropy terms is concave.";
                return true;
            }
        }

        return false;
    }
};

// ----------------- KL Divergence Detector -----------------------------------
// x*log(x/y) - x + y = x*log(x) - x*log(y) - x + y
// Standard form: sum_i (x_i * log(x_i / y_i) - x_i + y_i)
// This is convex when x, y > 0
template <class GraphT, class NodeT>
struct KLDetector final : ISuspectDetector<GraphT, NodeT> {
    using T = SuspectTraits<GraphT, NodeT>;
    bool detect(const GraphT&, const NodeT& n,
                const std::vector<NodeInfo>& kids, NodeInfo& info) override {
        auto op = T::op(n);
        auto ins = T::inputs(n);

        // KL term: a * log(a / b) - a + b  (where a, b are positive)
        // Detect: x * log(x / y) + (-x) + y  (n-ary Add)
        if (op != T::Op::Add) return false;
        if (ins.size() < 3) return false; // need at least 3 terms

        // KL: a*log(a/b) - a + b is convex when a,b > 0
        // Detect a*log(a/b) term: a * log(a / b)
        auto is_kl_term = [&](const NodeT* t) -> bool {
            if (!t || T::op(t) != T::Op::Multiply || t->inputs.size() < 2)
                return false;
            for (size_t i = 0; i < t->inputs.size(); ++i) {
                auto a = t->inputs[i];
                auto rest = t->inputs[1 - i];
                if (T::op(rest) != T::Op::Log || rest->inputs.size() != 1)
                    continue;
                if (T::op(rest->inputs[0]) != T::Op::Divide ||
                    rest->inputs[0].inputs.size() != 2) continue;
                // rest = log(a/b)
                if (rest->inputs[0].inputs[0].get() == a.get())
                    return true;
            }
            return false;
        };

        int n_kl_terms = 0;
        for (const auto& term : ins) {
            if (is_kl_term(term.get()))
                ++n_kl_terms;
        }
        if (n_kl_terms >= 1) {
            info.cvx = CvxTag::Convex;
            info.hint = "KL divergence form x*log(x/y) is convex.";
            return true;
        }

        // Also detect relative_entropy(a||b) = a*log(a/b) - a + b
        // as a sum with a*log(a/b) + (-a) + b
        auto is_neg_linear = [](const NodeT* t) -> bool {
            if (!t || T::op(t) != T::Op::Multiply) return false;
            for (auto& inp : t->inputs) {
                if (T::is_const(inp) && std::abs(T::const_value(inp) + 1.0) < 1e-12)
                    return true;
            }
            return false;
        };
        auto is_pos_linear = [](const NodeT* t) -> bool {
            if (!t || T::op(t) != T::Op::Multiply) return false;
            for (auto& inp : t->inputs) {
                if (T::is_const(inp) && std::abs(T::const_value(inp) - 1.0) < 1e-12)
                    return true;
            }
            return false;
        };

        bool has_neg = false, has_pos = false;
        for (const auto& term : ins) {
            if (is_kl_term(term.get())) { n_kl_terms++; }
            else if (is_neg_linear(term.get())) { has_neg = true; }
            else if (is_pos_linear(term.get())) { has_pos = true; }
        }
        if (n_kl_terms >= 1 && has_neg && has_pos) {
            info.cvx = CvxTag::Convex;
            info.hint = "KL form x*log(x/y)-x+y is convex.";
            return true;
        }

        return false;
    }
};

// ----------------- Relative Entropy Detector --------------------------------
// relative_entropy(a||b) = a*log(a/b) - a + b (convex for a,b > 0)
// This is the same as KL, so KLDetector covers it. We register KLDetector here.

// --------------------------- Analyzer ---------------------------------------
template <class GraphT, class NodeT>
class Analyzer {
    using T = SuspectTraits<GraphT, NodeT>;

public:
    struct Summary {
        bool is_convex_problem{false};

        struct Obj {
            CvxTag   cvx{CvxTag::Affine};
            MonoTag  mono{MonoTag::Unknown};
            Interval bounds{};
        } objective;

        struct VarRec {
            std::string name;
            size_t      index;
            Interval    bounds;
            bool        is_binary{false};
            CvxTag      cvx;
            MonoTag     mono;
        };
        std::vector<VarRec> variables;

        struct ConRec {
            size_t      index;
            char        sense;
            CvxTag      cvx;
            MonoTag     mono;
            Interval    bounds;
            bool        is_convex_constraint{false};
            std::string hint;
        };
        std::vector<ConRec> constraints;
    };

    explicit Analyzer(const GraphT &G) : G_(G) {
        // Built-in detectors
        registerDefaultDetectors_();
        buildTopo_();
        initLeafs_();
    }

    void registerDetector(std::shared_ptr<ISuspectDetector<GraphT, NodeT>> det) {
        detectors_.push_back(std::move(det));
    }

    const std::unordered_map<size_t, NodeInfo> &analyze(int max_iter = 10, double tol = 1e-12) {
        (void)tol;
        for (int it = 0; it < max_iter; ++it) {
            changed_.clear();
            bool fwd = forward_();
            bool bwd = backward_();
            if (!fwd && !bwd) break;
        }
        // Ensure mono/cvx set
        for (auto id : topo_) {
            auto        &ni = info_.at(id);
            const NodeT &n  = *nodes_by_id_.at(id);
            if (ni.mono == MonoTag::Unknown) ni.mono = evalMono_(n);
            if (ni.cvx == CvxTag::Unknown) ni.cvx = evalCvx_(n);
        }
        return info_;
    }

    const NodeInfo *get(size_t node_id) const {
        auto it = info_.find(node_id);
        return (it == info_.end()) ? nullptr : &it->second;
    }

    Summary buildSummary(bool print_summary = false) const {
        Summary S;

        // Objective
        if (T::has_objective(G_)) {
            const NodeT obj = T::objective(G_);
            if (const auto *oi = get(T::id(obj))) {
                S.objective.cvx    = oi->cvx;
                S.objective.mono   = oi->mono;
                S.objective.bounds = oi->bounds;
            } else {
                if (obj) {
                    S.objective.cvx  = CvxTag::Affine;
                    S.objective.mono = MonoTag::Unknown;
                    if (T::is_const(obj)) {
                        S.objective.bounds = Interval::Point(T::const_value(obj));
                    } else if (T::is_var(obj)) {
                        S.objective.bounds = Interval(T::var_lb(obj), T::var_ub(obj));
                    }
                }
            }
        }

        // Variables (from analyzed graph)
        auto vars = T::variables(G_);
        S.variables.reserve(vars.size());
        for (auto &v : vars) {
            const auto *vi = get(T::id(v));
            NodeInfo    z  = vi ? *vi : NodeInfo{};
            S.variables.push_back({T::var_name(v), T::var_index(v), z.bounds, T::is_binary(v), z.cvx, z.mono});
        }

        // Constraints
        bool all_ok = true;
        auto cons   = T::constraints(G_);
        S.constraints.reserve(cons.size());
        for (size_t i = 0; i < cons.size(); ++i) {
            const auto &c  = cons[i];
            const auto *ci = get(T::id(c.expr));
            NodeInfo    z  = ci ? *ci : NodeInfo{};

            bool        ok = false;
            std::string hint;

            if (c.sense == '<') {
                ok = (z.cvx == CvxTag::Convex || z.cvx == CvxTag::Affine);
                if (!ok && z.cvx == CvxTag::Concave) {
                    hint = "Concave ≤ b: multiply by -1 to get convex ≥ -b.";
                } else if (!ok) {
                    hint = "Non-convex constraint: consider reformulation.";
                }
            } else if (c.sense == '=') {
                ok = (z.cvx == CvxTag::Affine);
                if (!ok) { hint = "Equality with non-affine expression is non-convex."; }
            } else if (c.sense == '>') {
                ok = (z.cvx == CvxTag::Concave || z.cvx == CvxTag::Affine);
                if (!ok && z.cvx == CvxTag::Convex) {
                    hint = "Convex ≥ b: multiply by -1 to get concave ≤ -b.";
                } else if (!ok) {
                    hint = "Non-convex constraint: consider reformulation.";
                }
            }

            if (!ok && !z.hint.empty()) hint = z.hint;

            S.constraints.push_back({i, c.sense, z.cvx, z.mono, z.bounds, ok, hint});
            all_ok = all_ok && ok;
        }

        // Overall convexity check
        const int  obj_sense = T::objective_sense(G_);
        const bool obj_ok    = (obj_sense >= 0)
                                   ? (S.objective.cvx == CvxTag::Convex || S.objective.cvx == CvxTag::Affine)
                                   : (S.objective.cvx == CvxTag::Concave || S.objective.cvx == CvxTag::Affine);
        S.is_convex_problem  = obj_ok && all_ok;

        if (print_summary) printSummary_(S);
        return S;
    }

private:
    const GraphT                                                 &G_;
    std::vector<NodeT>                                            nodes_;
    std::unordered_map<size_t, const NodeT *>                     nodes_by_id_;
    std::vector<size_t>                                           topo_;
    std::unordered_map<size_t, NodeInfo>                          info_;
    std::unordered_set<size_t>                                    changed_;
    std::vector<std::shared_ptr<ISuspectDetector<GraphT, NodeT>>> detectors_;

    void registerDefaultDetectors_() {
        registerDetector(std::make_shared<SocNormDetector<GraphT, NodeT>>());
        registerDetector(std::make_shared<QuadraticSimpleDetector<GraphT, NodeT>>());
        registerDetector(std::make_shared<FractionalAffineDetector<GraphT, NodeT>>());
        registerDetector(std::make_shared<QuadOverLinDetector<GraphT, NodeT>>());
        registerDetector(std::make_shared<LogSumExpDetector<GraphT, NodeT>>());
        registerDetector(std::make_shared<EntropyDetector<GraphT, NodeT>>());
        registerDetector(std::make_shared<KLDetector<GraphT, NodeT>>());
    }

    // ------------------- Topological sort (parents adjacency) -------------------
    void buildTopo_() {
        nodes_ = T::nodes(G_);
        nodes_by_id_.clear();
        nodes_by_id_.reserve(nodes_.size());
        for (auto &n : nodes_) nodes_by_id_[T::id(n)] = &n;

        // Build reverse adjacency: input -> vector of parents that consume it
        std::unordered_map<size_t, std::vector<size_t>> parents_of;
        parents_of.reserve(nodes_.size());

        // In-degree = number of inputs
        std::unordered_map<size_t, int> indeg;
        indeg.reserve(nodes_.size());
        for (auto &n : nodes_) {
            const size_t nid = T::id(n);
            const auto  &ins = T::inputs(n);
            indeg[nid] = static_cast<int>(ins.size());
            for (auto &c : ins) parents_of[T::id(c)].push_back(nid); // c -> n
        }

        std::vector<size_t> q;
        q.reserve(nodes_.size());
        for (auto &n : nodes_) if (indeg[T::id(n)] == 0) q.push_back(T::id(n)); // leaves

        topo_.clear();
        topo_.reserve(nodes_.size());
        while (!q.empty()) {
            size_t v = q.back();
            q.pop_back();
            topo_.push_back(v);

            auto it = parents_of.find(v);
            if (it == parents_of.end()) continue;

            for (size_t parent_id : it->second) {
                if (--indeg[parent_id] == 0) q.push_back(parent_id);
            }
        }
        // Optional: detect cycles if topo_.size() < nodes_.size()
    }

    void initLeafs_() {
        info_.clear();
        info_.reserve(nodes_.size());
        for (auto id : topo_) {
            const NodeT &n = *nodes_by_id_[id];
            NodeInfo     ni;
            if (T::is_const(n)) {
                ni.bounds      = Interval::Point(T::const_value(n));
                ni.cvx         = CvxTag::Affine;
                ni.mono        = MonoTag::Unknown;
                ni.is_linear   = true;
                ni.poly_degree = 0;
            } else if (T::is_var(n)) {
                double lb = T::var_lb(n), ub = T::var_ub(n);
                ni.bounds      = Interval(lb, ub);
                ni.cvx         = CvxTag::Affine;
                ni.mono        = MonoTag::Unknown;
                ni.is_linear   = true;
                ni.poly_degree = 1;
            }
            info_.emplace(id, std::move(ni));
        }
    }

    bool forward_() {
        bool any = false;
        for (auto id : topo_) {
            const NodeT& n = *nodes_by_id_[id];
            NodeInfo& ni   = info_.at(id);

            Interval oldB = ni.bounds;
            MonoTag  oldM = ni.mono;
            CvxTag   oldC = ni.cvx;

            Interval nb = evalBounds_(n);
            MonoTag  nm = evalMono_(n);
            CvxTag   nc = evalCvx_(n);

            if (!oldB.empty()) nb = oldB.intersect(nb);
            if (nb.empty() && !oldB.empty()) nb = oldB;

            if (nb.lo != oldB.lo || nb.hi != oldB.hi) { ni.bounds = nb; changed_.insert(id); any = true; }
            if (nm != oldM) { ni.mono = nm; changed_.insert(id); any = true; }
            if (nc != oldC) { ni.cvx  = nc; changed_.insert(id); any = true; }
        }
        return any;
    }

    bool backward_() {
        bool any = false;
        for (auto it = topo_.rbegin(); it != topo_.rend(); ++it) {
            size_t id = *it;
            if (!changed_.count(id)) continue;

            const NodeT &p   = *nodes_by_id_[id];
            const auto   op  = T::op(p);
            const auto   ch  = T::inputs(p);
            auto         chN = ch.size();

            if (chN == 1) {
                if (backpropUnary_(p, ch[0])) { any = true; changed_.insert(T::id(ch[0])); }
            } else if (chN == 2) {
                if (backpropBinaryLeft_(p, ch[0], ch[1]))  { any = true; changed_.insert(T::id(ch[0])); }
                if (backpropBinaryRight_(p, ch[0], ch[1])) { any = true; changed_.insert(T::id(ch[1])); }
            } else if (chN >= 2) {
                // Linear n-ary backprop (sum)
                if (op == T::Op::Add) {
                    if (backpropLinearNary_(p, ch)) any = true;
                }
                // Max/Min
                if (op == T::Op::Max || op == T::Op::Min) {
                    const Interval PB = info_.at(id).bounds;
                    if (!PB.empty()) {
                        for (auto &a : ch) {
                            auto    &ci = info_.at(T::id(a));
                            Interval nb = ci.bounds;
                            if (op == T::Op::Max) nb = Interval(nb.lo, std::min(nb.hi, PB.hi));
                            else                  nb = Interval(std::max(nb.lo, PB.lo), nb.hi);
                            if (nb.lo != ci.bounds.lo || nb.hi != ci.bounds.hi) {
                                ci.bounds = nb;
                                any = true;
                                changed_.insert(T::id(a));
                            }
                        }
                    }
                }
            }
        }
        return any;
    }

    // New: linear n-ary tightening for Add (sum)
    bool backpropLinearNary_(const NodeT &parent, const std::vector<NodeT> &ch) {
        auto &P = info_.at(T::id(parent));
        if (P.bounds.empty()) return false;

        bool changed = false;

        const size_t k = ch.size();
        std::vector<Interval> xb(k);
        for (size_t i = 0; i < k; ++i) {
            xb[i] = info_.at(T::id(ch[i])).bounds;
            if (xb[i].empty()) return false;
        }

        // y = sum_i x_i  ->  for each j: x_j ∈ (y - sum_{i≠j} x_i) ∩ x_j
        for (size_t j = 0; j < k; ++j) {
            Interval S{0.0, 0.0};
            for (size_t i = 0; i < k; ++i) if (i != j) {
                S = S.add(xb[i]);
            }
            Interval rhs = P.bounds.add(S.neg());
            Interval newXj = rhs.intersect(xb[j]);
            auto &CJ = info_.at(T::id(ch[j])).bounds;
            if (newXj.lo != CJ.lo || newXj.hi != CJ.hi) {
                CJ = newXj; changed = true; changed_.insert(T::id(ch[j]));
            }
        }
        return changed;
    }

    Interval evalBounds_(const NodeT &n) {
        using Op = typename T::Op;
        auto op  = T::op(n);

        if (T::is_const(n)) return Interval::Point(T::const_value(n));
        if (T::is_var(n))   return Interval(T::var_lb(n), T::var_ub(n));

        auto ins = T::inputs(n);
        auto get = [&](int i) { return info_.at(T::id(ins[size_t(i)])).bounds; };

        switch (op) {
        case Op::cte: return Interval::Point(T::const_value(n));
        case Op::Neg: return get(0).neg();
        case Op::Add: {
            Interval acc = Interval::Point(0.0);
            for (size_t i = 0; i < ins.size(); ++i)
                acc = acc.add(get(static_cast<int>(i)));
            return acc;
        }
        case Op::Subtract: return get(0).add(get(1).neg());
        case Op::Multiply: {
            if (ins.size() == 2 && T::id(ins[0]) == T::id(ins[1]))
                return get(0).pow(2.0);
            if (ins.empty()) return Interval::Point(1.0);
            Interval acc = get(0);
            for (size_t i = 1; i < ins.size(); ++i)
                acc = acc.mul(get(static_cast<int>(i)));
            return acc;
        }
        case Op::Divide:   return get(0).div(get(1));
        case Op::Pow:      return get(0).pow(T::pow_exponent(n));
        case Op::Abs: {
            auto a = get(0);
            if (a.empty()) return Interval::Empty();
            double m = std::max(std::abs(a.lo), std::abs(a.hi));
            return Interval(0.0, m);
        }
        case Op::Exp: return get(0).exp();
        case Op::Log: {
            auto a = get(0);
            if (a.hi <= 0.0) return Interval::Empty();
            return a.log();
        }
        case Op::Sqrt: {
            auto a = get(0);
            if (a.hi < 0.0) return Interval::Empty();
            return a.sqrt();
        }
        case Op::Sin:
        case Op::Cos: return Interval(-1.0, 1.0);
        case Op::Tan: return Interval::Full();
        case Op::Max: {
            if (ins.empty()) return Interval::Full();
            Interval acc = info_.at(T::id(ins[0])).bounds;
            for (size_t i = 1; i < ins.size(); ++i) {
                auto b = info_.at(T::id(ins[i])).bounds;
                if (!acc.empty() && !b.empty())
                    acc = Interval(std::max(acc.lo, b.lo), std::max(acc.hi, b.hi));
            }
            return acc;
        }
        case Op::Min: {
            if (ins.empty()) return Interval::Full();
            Interval acc = info_.at(T::id(ins[0])).bounds;
            for (size_t i = 1; i < ins.size(); ++i) {
                auto b = info_.at(T::id(ins[i])).bounds;
                if (!acc.empty() && !b.empty())
                    acc = Interval(std::min(acc.lo, b.lo), std::min(acc.hi, b.hi));
            }
            return acc;
        }
        default: return Interval::Full();
        }
    }

    static MonoTag negate(MonoTag m) {
        switch (m) {
        case MonoTag::Nondecreasing: return MonoTag::Nonincreasing;
        case MonoTag::Nonincreasing: return MonoTag::Nondecreasing;
        default: return MonoTag::Unknown;
        }
    }
    static CvxTag negate(CvxTag c) {
        switch (c) {
        case CvxTag::Convex: return CvxTag::Concave;
        case CvxTag::Concave: return CvxTag::Convex;
        case CvxTag::Affine: return CvxTag::Affine;
        default: return CvxTag::Unknown;
        }
    }
    static CvxTag addCvx(CvxTag a, CvxTag b) {
        if (a == CvxTag::Unknown || b == CvxTag::Unknown) return CvxTag::Unknown;
        if (a == CvxTag::Affine) return b;
        if (b == CvxTag::Affine) return a;
        return (a == b) ? a : CvxTag::Unknown;
    }
    static bool isConvexLike(CvxTag c) {
        return c == CvxTag::Convex || c == CvxTag::Affine;
    }
    static bool isConcaveLike(CvxTag c) {
        return c == CvxTag::Concave || c == CvxTag::Affine;
    }

    MonoTag evalMono_(const NodeT &n) {
        using Op = typename T::Op;
        auto op  = T::op(n);
        auto ins = T::inputs(n);
        if (ins.empty()) return MonoTag::Unknown;

        auto mono = [&](int i) { return info_.at(T::id(ins[size_t(i)])).mono; };
        auto bnds = [&](int i) { return info_.at(T::id(ins[size_t(i)])).bounds; };

        switch (op) {
        case Op::Var: return MonoTag::Nondecreasing;
        case Op::cte: return MonoTag::Unknown;
        case Op::Neg: return negate(mono(0));
        case Op::Add: {
            bool all_nd = true, all_ni = true;
            for (size_t i = 0; i < ins.size(); ++i) {
                MonoTag mi = mono(static_cast<int>(i));
                all_nd = all_nd && (mi == MonoTag::Nondecreasing);
                all_ni = all_ni && (mi == MonoTag::Nonincreasing);
            }
            if (all_nd) return MonoTag::Nondecreasing;
            if (all_ni) return MonoTag::Nonincreasing;
            return MonoTag::Unknown;
        }
        case Op::Subtract: {
            MonoTag a = mono(0), b = negate(mono(1));
            if (a == MonoTag::Nondecreasing && b == MonoTag::Nondecreasing) return MonoTag::Nondecreasing;
            if (a == MonoTag::Nonincreasing && b == MonoTag::Nondecreasing) return MonoTag::Nonincreasing;
            return MonoTag::Unknown;
        }
        case Op::Multiply: {
            if (T::is_const(ins[0])) {
                double c = T::const_value(ins[0]);
                if (c > 0) return mono(1);
                if (c < 0) return negate(mono(1));
                return MonoTag::Unknown;
            }
            if (T::is_const(ins[1])) {
                double c = T::const_value(ins[1]);
                if (c > 0) return mono(0);
                if (c < 0) return negate(mono(0));
                return MonoTag::Unknown;
            }
            if (bnds(0).nonnegative() && bnds(1).nonnegative()) {
                if (mono(0) == mono(1)) return mono(0);
            }
            return MonoTag::Unknown;
        }
        case Op::Divide: {
            if (T::is_const(ins[1])) {
                double c = T::const_value(ins[1]);
                if (c > 0) return mono(0);
                if (c < 0) return negate(mono(0));
            }
            return MonoTag::Unknown;
        }
        case Op::Pow: {
            double p = T::pow_exponent(n);
            auto   B = bnds(0);
            if (std::abs(p - 1.0) < 1e-12) return mono(0);
            if (p > 0 && B.nonnegative()) return mono(0);
            if (p < 0 && B.positive()) return negate(mono(0));
            return MonoTag::Unknown;
        }
        case Op::Abs: {
            auto B = bnds(0);
            if (B.nonnegative()) return mono(0);
            if (B.nonpositive()) return negate(mono(0));
            return MonoTag::Unknown;
        }
        case Op::Exp:
        case Op::Log:
        case Op::Sqrt: return mono(0);
        case Op::Sin:
        case Op::Cos:
        case Op::Tan: return MonoTag::Unknown;
        case Op::Max:
        case Op::Min: {
            if (ins.empty()) return MonoTag::Unknown;
            MonoTag m = info_.at(T::id(ins[0])).mono;
            for (size_t i = 1; i < ins.size(); ++i) if (info_.at(T::id(ins[i])).mono != m) return MonoTag::Unknown;
            return m;
        }
        default: return MonoTag::Unknown;
        }
    }

    CvxTag evalCvx_(const NodeT &n) {
        using Op = typename T::Op;
        auto op  = T::op(n);

#ifndef NDEBUG
        if (!T::is_const(n) && !T::is_var(n)) {
            std::cout << "evalCvx: node " << T::id(n) << " op=" << static_cast<int>(op)
                      << " #inputs=" << T::inputs(n).size() << "\n";
        }
#endif

        if (T::is_const(n) || T::is_var(n)) return CvxTag::Affine;

        auto ins = T::inputs(n);
        if (ins.empty()) return CvxTag::Unknown;

        auto cvx = [&](int i) { return info_.at(T::id(ins[size_t(i)])).cvx; };
        auto bnd = [&](int i) { return info_.at(T::id(ins[size_t(i)])).bounds; };

        NodeInfo &ni = info_.at(T::id(n));

        auto finish_with_plugins = [&](CvxTag tag) -> CvxTag {
            ni.cvx = tag;
            if (!detectors_.empty()) {
                std::vector<NodeInfo> childInfos;
                childInfos.reserve(ins.size());
                for (auto &c : ins) childInfos.push_back(info_.at(T::id(c)));
                for (auto &det : detectors_) {
                    if (det->detect(G_, n, childInfos, ni)) return ni.cvx;
                }
            }
            return ni.cvx;
        };

        switch (op) {
        case Op::Neg: return finish_with_plugins(negate(cvx(0)));
        case Op::Add: {
            CvxTag acc = CvxTag::Affine;
            for (size_t i = 0; i < ins.size(); ++i)
                acc = addCvx(acc, cvx(static_cast<int>(i)));
            return finish_with_plugins(acc);
        }
        case Op::Subtract: return finish_with_plugins(addCvx(cvx(0), negate(cvx(1))));
        case Op::Abs:
            if (bnd(0).nonnegative()) {
                ni.hint = "Abs reduces to identity on nonnegative domain.";
                return finish_with_plugins(cvx(0));
            }
            if (bnd(0).nonpositive()) {
                ni.hint = "Abs reduces to negation on nonpositive domain.";
                return finish_with_plugins(negate(cvx(0)));
            }
            if (cvx(0) == CvxTag::Affine) {
                ni.hint = "Abs is convex for affine arguments.";
                return finish_with_plugins(CvxTag::Convex);
            }
            ni.hint = "Abs is only DCP-safe here for affine or sign-fixed arguments.";
            return finish_with_plugins(CvxTag::Unknown);
        case Op::Exp:
            if (isConvexLike(cvx(0))) {
                ni.hint = "Exp is convex and increasing, so convex/affine input is allowed.";
                return finish_with_plugins(CvxTag::Convex);
            }
            ni.hint = "Exp requires convex/affine input under DCP composition.";
            return finish_with_plugins(CvxTag::Unknown);
        case Op::Log:
            if (bnd(0).positive() && (cvx(0) == CvxTag::Concave || cvx(0) == CvxTag::Affine)) {
                ni.hint = "Log is concave on positive domain.";
                return finish_with_plugins(CvxTag::Concave);
            }
            ni.hint = "Log requires positive concave/affine argument.";
            return finish_with_plugins(CvxTag::Unknown);
        case Op::Sqrt:
            if (bnd(0).nonnegative() && (cvx(0) == CvxTag::Concave || cvx(0) == CvxTag::Affine)) {
                ni.hint = "Sqrt is concave on nonnegative domain.";
                return finish_with_plugins(CvxTag::Concave);
            }
            ni.hint = "Sqrt requires nonnegative concave/affine argument.";
            return finish_with_plugins(CvxTag::Unknown);
        case Op::Multiply: {
            if (ins.size() != 2) {
                int nonconst = 0, nonconst_idx = -1;
                double scale = 1.0;
                for (size_t i = 0; i < ins.size(); ++i) {
                    if (T::is_const(ins[i])) {
                        scale *= T::const_value(ins[i]);
                    } else {
                        nonconst++;
                        nonconst_idx = static_cast<int>(i);
                    }
                }
                if (nonconst == 0) return finish_with_plugins(CvxTag::Affine);
                if (nonconst == 1) {
                    CvxTag result =
                        (scale > 0.0) ? cvx(nonconst_idx)
                                      : (scale < 0.0) ? negate(cvx(nonconst_idx))
                                                      : CvxTag::Affine;
                    return finish_with_plugins(result);
                }
                ni.hint = "N-ary product is only certified when all but one factors are constant.";
                return finish_with_plugins(CvxTag::Unknown);
            }
            if (T::is_const(ins[0])) {
                double c      = T::const_value(ins[0]);
                CvxTag result = (c > 0) ? cvx(1) : (c < 0) ? negate(cvx(1)) : CvxTag::Affine;
                return finish_with_plugins(result);
            }
            if (T::is_const(ins[1])) {
                double c      = T::const_value(ins[1]);
                CvxTag result = (c > 0) ? cvx(0) : (c < 0) ? negate(cvx(0)) : CvxTag::Affine;
                return finish_with_plugins(result);
            }
            if (T::id(ins[0]) == T::id(ins[1])) {
                if (cvx(0) == CvxTag::Affine) {
                    ni.hint = "x*x = x^2 is convex for affine x.";
                    return finish_with_plugins(CvxTag::Convex);
                }
                if (bnd(0).nonnegative() && isConvexLike(cvx(0))) {
                    ni.hint = "x^2 is convex for nonnegative convex input.";
                    return finish_with_plugins(CvxTag::Convex);
                }
                if (bnd(0).nonpositive() && isConcaveLike(cvx(0))) {
                    ni.hint = "x^2 is convex for nonpositive concave input.";
                    return finish_with_plugins(CvxTag::Convex);
                }
                ni.hint = "x*x is only certified for affine or sign-compatible input.";
                return finish_with_plugins(CvxTag::Unknown);
            }
            ni.hint = "Bilinear term: consider McCormick/perspective.";
            return finish_with_plugins(CvxTag::Unknown);
        }
        case Op::Divide: {
            if (T::is_const(ins[1])) {
                double c      = T::const_value(ins[1]);
                CvxTag result = (c > 0) ? cvx(0) : (c < 0) ? negate(cvx(0)) : CvxTag::Unknown;
                return finish_with_plugins(result);
            }
            if (T::is_const(ins[0]) && bnd(1).positive() && (cvx(1) == CvxTag::Concave || cvx(1)==CvxTag::Affine)) {
                double c      = T::const_value(ins[0]);
                CvxTag result = (c >= 0.0) ? CvxTag::Convex : CvxTag::Concave;
                ni.hint       = "c/concave with positive denominator.";
                return finish_with_plugins(result);
            }
            ni.hint = "General quotient: check DCP or add auxiliaries.";
            return finish_with_plugins(CvxTag::Unknown);
        }
        case Op::Pow: {
            double p = T::pow_exponent(n);
            if (std::abs(p - 1.0) < 1e-12) return finish_with_plugins(cvx(0));
            if (std::abs(p) < 1e-12)       return finish_with_plugins(CvxTag::Affine);
            if (std::abs(p - 2.0) < 1e-12) {
                if (cvx(0) == CvxTag::Affine) { ni.hint = "x^2 convex for affine x."; return finish_with_plugins(CvxTag::Convex); }
                ni.hint = "f(x)^2 convex only if f affine.";
                return finish_with_plugins(CvxTag::Unknown);
            }
            if (p >= 1.0 && bnd(0).nonnegative() && (cvx(0) == CvxTag::Convex || cvx(0) == CvxTag::Affine)) {
                ni.hint = "x^p convex for p>=1, x>=0, convex/affine x.";
                return finish_with_plugins(CvxTag::Convex);
            }
            if (p > 0.0 && p < 1.0 && bnd(0).positive() && (cvx(0) == CvxTag::Concave || cvx(0) == CvxTag::Affine)) {
                ni.hint = "x^p concave for 0<p<1, x>0, concave/affine x.";
                return finish_with_plugins(CvxTag::Concave);
            }
            ni.hint = "Power: check domain and DCP rules.";
            return finish_with_plugins(CvxTag::Unknown);
        }
        case Op::Max: {
            bool ok = true;
            for (auto &a : ins) {
                if (!isConvexLike(info_.at(T::id(a)).cvx)) { ok = false; break; }
            }
            if (ok) {
                ni.hint = "Max of convex functions is convex.";
                return finish_with_plugins(CvxTag::Convex);
            }
            ni.hint = "Max requires convex/affine arguments.";
            return finish_with_plugins(CvxTag::Unknown);
        }
        case Op::Min: {
            bool ok = true;
            for (auto &a : ins) {
                CvxTag ca = info_.at(T::id(a)).cvx;
                if (!(ca == CvxTag::Concave || ca == CvxTag::Affine)) { ok = false; break; }
            }
            if (ok) { ni.hint = "Min of concave functions is concave."; return finish_with_plugins(CvxTag::Concave); }
            ni.hint = "Min requires concave/affine arguments.";
            return finish_with_plugins(CvxTag::Unknown);
        }
        default:
            ni.hint = "Unknown operator.";
            return finish_with_plugins(CvxTag::Unknown);
        }
    }

    bool backpropUnary_(const NodeT &parent, const NodeT &child) {
        auto &P = info_.at(T::id(parent));
        auto &C = info_.at(T::id(child));
        if (P.bounds.empty()) return false;

        Interval inv = Interval::Full();
        switch (T::op(parent)) {
        case T::Op::Neg:   inv = P.bounds.neg(); break;
        case T::Op::Abs:   inv = Interval(-P.bounds.hi, P.bounds.hi); break;
        case T::Op::Exp:   if (P.bounds.positive())
                               inv = Interval(std::log(std::max(P.bounds.lo, 1e-300)), std::log(P.bounds.hi));
                           break;
        case T::Op::Log:   inv = Interval(std::exp(P.bounds.lo), std::exp(P.bounds.hi)); break;
        case T::Op::Sqrt:  if (P.bounds.nonnegative())
                               inv = Interval(P.bounds.lo * P.bounds.lo, P.bounds.hi * P.bounds.hi);
                           break;
        default: break;
        }

        Interval New = C.bounds.intersect(inv);
        if (New.lo != C.bounds.lo || New.hi != C.bounds.hi) { C.bounds = New; return true; }
        return false;
    }

    bool backpropBinaryLeft_(const NodeT &parent, const NodeT &left, const NodeT &right) {
        auto &P = info_.at(T::id(parent));
        auto &L = info_.at(T::id(left));
        auto &R = info_.at(T::id(right));
        if (P.bounds.empty() || R.bounds.empty()) return false;

        Interval inv = Interval::Full();
        switch (T::op(parent)) {
        case T::Op::Add:      inv = P.bounds.add(R.bounds.neg()); break;
        case T::Op::Subtract: inv = P.bounds.add(R.bounds); break;
        case T::Op::Multiply: inv = P.bounds.div(R.bounds); break;
        case T::Op::Divide:   inv = P.bounds.mul(R.bounds); break;
        default: break;
        }

        Interval New = L.bounds.intersect(inv);
        if (New.lo != L.bounds.lo || New.hi != L.bounds.hi) { L.bounds = New; return true; }
        return false;
    }

    bool backpropBinaryRight_(const NodeT &parent, const NodeT &left, const NodeT &right) {
        auto &P = info_.at(T::id(parent));
        auto &L = info_.at(T::id(left));
        auto &R = info_.at(T::id(right));
        if (P.bounds.empty() || L.bounds.empty()) return false;

        Interval inv = Interval::Full();
        switch (T::op(parent)) {
        case T::Op::Add:      inv = P.bounds.add(L.bounds.neg()); break;
        case T::Op::Subtract: inv = L.bounds.add(P.bounds.neg()); break;
        case T::Op::Multiply: inv = P.bounds.div(L.bounds); break;
        case T::Op::Divide:
            if (!P.bounds.containsZero() && !L.bounds.containsZero()) { inv = L.bounds.div(P.bounds); }
            break;
        default: break;
        }

        Interval New = R.bounds.intersect(inv);
        if (New.lo != R.bounds.lo || New.hi != R.bounds.hi) { R.bounds = New; return true; }
        return false;
    }

    void printSummary_(const Summary &S) const {
        const int obj_sense = T::objective_sense(G_);
        std::cout << "=== SUSPECT Convexity Analysis ===\n\n";

        // Objective
        std::cout << "[Objective]\n";
        if (obj_sense != 0) {
            std::cout << "  Type:         " << (obj_sense > 0 ? "minimize" : "maximize") << "\n";
            std::cout << "  Convexity:    " << to_string(S.objective.cvx) << "\n";
            std::cout << "  Monotonicity: " << to_string(S.objective.mono) << "\n";
            std::cout << "  Bounds:       " << S.objective.bounds.to_string() << "\n";
            bool obj_ok = (obj_sense >= 0)
                            ? (S.objective.cvx == CvxTag::Convex || S.objective.cvx == CvxTag::Affine)
                            : (S.objective.cvx == CvxTag::Concave || S.objective.cvx == CvxTag::Affine);
            std::cout << "  Status:       " << (obj_ok ? "convex" : "non-convex") << "\n\n";
        } else {
            std::cout << "  (no objective)\n\n";
        }

        // Variables
        std::cout << "[Variables] (" << S.variables.size() << ")\n";
        if (S.variables.size() <= 30) {
            for (const auto &v : S.variables) {
                std::cout << "  - " << v.name << " [idx=" << v.index << "]  "
                          << (v.is_binary ? "binary" : to_string(v.cvx)) << "  "
                          << v.bounds.to_string() << "\n";
            }
        } else {
            size_t bin = 0;
            for (const auto &v : S.variables) bin += v.is_binary ? 1 : 0;
            std::cout << "  total=" << S.variables.size()
                      << "  binary=" << bin
                      << "  continuous=" << (S.variables.size()-bin) << "\n";
        }
        std::cout << "\n";

        // Constraints
        std::cout << "[Constraints] (" << S.constraints.size() << ")\n";
        size_t convex_count=0, affine_count=0, nonconvex_count=0;
        for (const auto &c : S.constraints) {
            if (c.is_convex_constraint) {
                if (c.cvx == CvxTag::Affine) affine_count++; else convex_count++;
            } else {
                nonconvex_count++;
            }
        }
        std::cout << "  convex/concave: " << convex_count << "\n";
        std::cout << "  affine:         " << affine_count << "\n";
        std::cout << "  non-convex:     " << nonconvex_count << "\n";
        if (S.constraints.size() <= 20 || nonconvex_count > 0) {
            for (const auto &c : S.constraints) {
                std::cout << "  - [" << c.index << "] "
                          << (c.is_convex_constraint ? "OK " : "NO ")
                          << to_string(c.cvx)
                          << "   sense=" << c.sense
                          << "   expr-bounds=" << c.bounds.to_string();
                if (!c.is_convex_constraint && !c.hint.empty())
                    std::cout << "\n      hint: " << c.hint;
                std::cout << "\n";
            }
        }
        std::cout << "\n";

        // Verdict
        std::cout << "[Verdict]\n";
        std::cout << "  Problem is " << (S.is_convex_problem ? "CONVEX" : "NON-CONVEX") << ".\n";
        if (!S.is_convex_problem) {
            bool any_hint = false;
            for (const auto &c : S.constraints)
                if (!c.is_convex_constraint && !c.hint.empty()) { any_hint = true; break; }
            if (any_hint) {
                std::cout << "  Recommendations:\n";
                for (const auto &c : S.constraints) {
                    if (!c.is_convex_constraint && !c.hint.empty())
                        std::cout << "    - Constraint " << c.index << ": " << c.hint << "\n";
                }
            } else {
                std::cout << "  Recommendations:\n"
                          << "    - Reformulate non-convex constraints where possible.\n"
                          << "    - Consider global optimization methods for remaining non-convexities.\n"
                          << "    - Look for bilinear/quadratic terms suitable for envelopes or perspectives.\n";
            }
        }
        std::cout << "\n";
    }
};

} // namespace suspect

// ============================================================================
// Adapter layer for AD graph
// ============================================================================
namespace suspect_adapter {

struct ExprView {
    ADGraphPtr             g;
    ADNodePtr              root;
    std::vector<ADNodePtr> vars;
    bool                   ok() const { return (bool)g && (bool)root; }
};
inline ExprView make_view(const GradFn &F) { return ExprView{F.g, F.expr_root, F.var_nodes}; }

struct ModelView {
    std::optional<ExprView>                objective;
    std::vector<std::pair<ExprView, char>> constraints;
    std::vector<ADNodePtr>                 all_nodes;
    std::vector<double>                    lb, ub;
    std::vector<bool>                      is_binary;
    int                                    obj_sense{+1};
};

struct VarMeta {
    int    index{-1};
    double lb{-std::numeric_limits<double>::infinity()};
    double ub{std::numeric_limits<double>::infinity()};
    bool   is_binary{false};
};
struct MetaRegistry { std::unordered_map<const ADNode *, VarMeta> meta; };
inline thread_local MetaRegistry g_meta;

inline void stamp_var_metadata(const std::vector<ADNodePtr> &var_nodes,
                               const Eigen::Ref<const Eigen::VectorXd> &lb,
                               const Eigen::Ref<const Eigen::VectorXd> &ub,
                               const std::vector<bool> &is_bin) {
    const std::size_t n = var_nodes.size();
    for (std::size_t i = 0; i < n; ++i) {
        const ADNode *key = var_nodes[i].get();
        auto         &m   = g_meta.meta[key];
        m.index           = static_cast<int>(i);
        m.lb              = lb(static_cast<int>(i));
        m.ub              = ub(static_cast<int>(i));
        m.is_binary       = (i < is_bin.size()) ? is_bin[i] : false;
    }
}

inline void collect_closure(ADNodePtr root, std::vector<ADNodePtr> &out) {
    if (!root) return;
    std::unordered_set<const ADNode *> seen;
    std::vector<ADNodePtr>             st{root};
    while (!st.empty()) {
        ADNodePtr u = st.back(); st.pop_back();
        if (!u) continue;
        if (seen.insert(u.get()).second) {
            out.push_back(u);
            for (auto &c : u->inputs) st.push_back(c);
        }
    }
}

// Ensure ModelView::all_nodes covers objective + ALL constraints
inline void populate_all_nodes(ModelView &M) {
    std::vector<ADNodePtr> acc;
    if (M.objective && M.objective->ok()) collect_closure(M.objective->root, acc);
    for (auto &c : M.constraints) if (c.first.ok()) collect_closure(c.first.root, acc);
    std::unordered_set<const ADNode*> seen;
    M.all_nodes.clear(); M.all_nodes.reserve(acc.size());
    for (auto &p : acc) if (p && seen.insert(p.get()).second) M.all_nodes.push_back(p);
}

// Const-safe fallbacks if all_nodes not populated
inline std::vector<ADNodePtr> collect_all_nodes_const(const ModelView &M) {
    std::vector<ADNodePtr> acc;
    if (M.objective && M.objective->ok()) collect_closure(M.objective->root, acc);
    for (auto &c : M.constraints) if (c.first.ok()) collect_closure(c.first.root, acc);
    std::unordered_set<const ADNode*> seen;
    std::vector<ADNodePtr> out; out.reserve(acc.size());
    for (auto &p : acc) if (p && seen.insert(p.get()).second) out.push_back(p);
    return out;
}
inline std::vector<ADNodePtr> collect_vars_from_nodes_const(const std::vector<ADNodePtr> &nodes) {
    std::unordered_set<const ADNode*> seen;
    std::vector<ADNodePtr> vars;
    for (auto &n : nodes) {
        if (!n || !n->isVariable()) continue;
        if (seen.insert(n.get()).second) vars.push_back(n);
    }
    std::stable_sort(vars.begin(), vars.end(), [](const ADNodePtr &a, const ADNodePtr &b){
        auto ia = suspect_adapter::g_meta.meta.find(a.get());
        auto ib = suspect_adapter::g_meta.meta.find(b.get());
        bool ha = (ia != suspect_adapter::g_meta.meta.end() && ia->second.index >= 0);
        bool hb = (ib != suspect_adapter::g_meta.meta.end() && ib->second.index >= 0);
        if (ha && hb) return ia->second.index < ib->second.index;
        if (ha != hb) return ha; // stamped first
        return a.get() < b.get();
    });
    return vars;
}

} // namespace suspect_adapter

// ============================================================================
// SuspectTraits specialization
// ============================================================================
namespace suspect {

template <>
struct SuspectTraits<suspect_adapter::ModelView, std::shared_ptr<ADNode>> {
    using GraphT = suspect_adapter::ModelView;
    using NodeT  = std::shared_ptr<ADNode>;
    using Op     = Operator;

    static std::vector<NodeT> nodes(const GraphT &G) {
        if (!G.all_nodes.empty()) return G.all_nodes;
        return suspect_adapter::collect_all_nodes_const(G);
    }
    static std::vector<NodeT> inputs(const NodeT &n) { return n->inputs; }
    static size_t             id(const NodeT &n) { return reinterpret_cast<size_t>(n.get()); }
    static Op                 op(const NodeT &n) { return n->type; }

    static bool   is_const(const NodeT &n) { return n->isConstant(); }
    static double const_value(const NodeT &n) { return n->value; }
    static bool   is_var(const NodeT &n) { return n->isVariable(); }

    static double var_lb(const NodeT &n) {
        auto it = suspect_adapter::g_meta.meta.find(n.get());
        return (it != suspect_adapter::g_meta.meta.end()) ? it->second.lb : -std::numeric_limits<double>::infinity();
    }
    static double var_ub(const NodeT &n) {
        auto it = suspect_adapter::g_meta.meta.find(n.get());
        return (it != suspect_adapter::g_meta.meta.end()) ? it->second.ub : std::numeric_limits<double>::infinity();
    }
    static bool is_binary(const NodeT &n) {
        auto it = suspect_adapter::g_meta.meta.find(n.get());
        if (it != suspect_adapter::g_meta.meta.end()) return it->second.is_binary;
        if (!n->isVariable()) return false;
        const double lb = var_lb(n), ub = var_ub(n);
        return std::isfinite(lb) && std::isfinite(ub) && std::abs(lb) <= 1e-12 && std::abs(ub - 1.0) <= 1e-12;
    }
    static std::string var_name(const NodeT &n) { return n->name; }
    static size_t var_index(const NodeT &n) {
        auto it = suspect_adapter::g_meta.meta.find(n.get());
        return (it != suspect_adapter::g_meta.meta.end() && it->second.index >= 0)
                   ? static_cast<size_t>(it->second.index)
                   : 0;
    }
    static double pow_exponent(const NodeT &n) {
        const auto &ins = n->inputs;
        if (ins.size() >= 2 && ins[1] && ins[1]->isConstant()) return ins[1]->value;
        return 1.0;
    }

    struct ConstraintRec { NodeT expr; char sense; };

    static bool has_objective(const GraphT &G) { return G.objective.has_value() && G.objective->ok(); }
    static NodeT objective(const GraphT &G) { return G.objective.has_value() ? G.objective->root : NodeT{}; }
    static int   objective_sense(const GraphT &G) { return G.obj_sense; }

    static std::vector<NodeT> variables(const GraphT &G) {
        auto N = nodes(G);
        auto V = suspect_adapter::collect_vars_from_nodes_const(N);
        if (!V.empty()) return V;
        if (G.objective.has_value()) return G.objective->vars;
        if (!G.constraints.empty())  return G.constraints.front().first.vars;
        return {};
    }

    static std::vector<ConstraintRec> constraints(const GraphT &G) {
        std::vector<ConstraintRec> out;
        out.reserve(G.constraints.size());
        for (const auto &c : G.constraints) if (c.first.ok()) out.push_back({c.first.root, c.second});
        return out;
    }
};

} // namespace suspect
