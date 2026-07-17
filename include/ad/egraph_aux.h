
// egraph_rules.hpp
#pragma once
#include <queue>

#include "ad_graph.h"
#include "ad/definitions.h"
#include "egraph.h"

// Forward decls from your codebase
struct ADNode;
using ADNodePtr = std::shared_ptr<ADNode>;
struct ADGraph;

// Small factory hooks you add to ADGraph (shown below patch)
ADNodePtr AD_makeNode(ADGraph& G, Operator op,
                      const std::vector<ADNodePtr>& kids);
ADNodePtr AD_makeConst(ADGraph& G, double v);

// Flatten + sort for AC ops
static void flatten_AC_(Operator op, const ADNodePtr& n,
                        std::vector<ADNodePtr>& out);

// Build EGraph from an AD subtree; returns e-class id
static int to_egraph_rec(const ADNodePtr& n, EGraph& EG) {
    // Constants
    if (n->type == Operator::cte) {
        ENode c;
        c.op = Operator::cte;
        c.is_const = true;
        c.cval = n->value;
        return EG.add(c);
    }

    // Terminals (variables / symbols) — use pointer identity
    if (n->inputs.empty() && n->type != Operator::cte) {
        ENode t;
        t.op = n->type;
        t.is_symbol = true;
        t.sym_ptr = (const void*)n.get();
        return EG.add(t);
    }

    // General case
    ENode e;
    e.op = n->type;
    std::vector<ADNodePtr> kids;
    if (n->type == Operator::Add || n->type == Operator::Multiply) {
        flatten_AC_(n->type, n, kids);  // n-ary flatten
        // translate each child
        for (auto& k : kids) e.kids.push_back({to_egraph_rec(k, EG)});
        // sort for AC canonical form
        std::sort(e.kids.begin(), e.kids.end(),
                  [](auto a, auto b) { return a.id < b.id; });
        return EG.add(e);
    } else {
        for (auto& k : n->inputs) e.kids.push_back({to_egraph_rec(k, EG)});
        return EG.add(e);
    }
}

inline void flatten_AC_(Operator op, const ADNodePtr& n,
                        std::vector<ADNodePtr>& out) {
    if (n->type == op) {
        for (auto& c : n->inputs) flatten_AC_(op, c, out);
    } else {
        out.push_back(n);
    }
}

// Extraction: pick cheapest enode in each class, then rebuild AD nodes
static ADNodePtr extract_rec(int eclass, const EGraph& EG, ECostModel& cm,
                             std::unordered_map<int, ADNodePtr>& memo,
                             ADGraph& G) {
    int arena_size = (int)EG.arena.size();
    int num_classes = (int)EG.classes.size();

    // Bounds check on eclass
    if (eclass < 0 || eclass >= num_classes) {
        return AD_makeConst(G, 0.0);
    }

    int rep = EG.find(eclass);
    // Bounds check on rep
    if (rep < 0 || rep >= num_classes) {
        return AD_makeConst(G, 0.0);
    }
    auto it = memo.find(rep);
    if (it != memo.end()) {
        return it->second;
    }

    // Safety: if the representative class is empty or invalid, return const 0
    if (EG.classes[rep].nodes.empty()) {
        auto r = AD_makeConst(G, 0.0);
        memo[rep] = r;
        return r;
    }

    std::unordered_map<int, int> costmemo;
    auto best = cm.best_of(rep, EG, costmemo);  // (cost, idx)
    if (best.second < 0 || best.second >= arena_size) {
        auto r = AD_makeConst(G, 0.0);
        memo[rep] = r;
        return r;
    }
    const ENode& n = EG.arena[best.second];

    if (n.is_const) {
        auto r = AD_makeConst(G, n.cval);
        memo[rep] = r;
        return r;
    }
    if (n.is_symbol) {
        ADNode* orig = (ADNode*)n.sym_ptr;
        ADNodePtr sptr(orig, [](ADNode*) {});  // aliasing, no ownership transfer
        memo[rep] = sptr;
        return sptr;
    }

    // Guard against infinite recursion in extract
    static thread_local int depth = 0;
    ++depth;
    if (depth > 100) { depth--; return AD_makeConst(G, 0.0); }

    std::vector<ADNodePtr> kids;
    kids.reserve(n.kids.size());
    for (auto k : n.kids) {
        // Bounds check on kid id
        int kid_ec = k.id;
        if (kid_ec < 0 || kid_ec >= num_classes) {
            kids.push_back(AD_makeConst(G, 0.0));
            continue;
        }
        kids.push_back(extract_rec(kid_ec, EG, cm, memo, G));
    }
    ADNodePtr built = AD_makeNode(G, n.op, kids);
    memo[rep] = built;
    depth--;
    return built;
}

struct EGraphBridge {
    static int to_egraph(const ADNodePtr& root, EGraph& EG) {
        return to_egraph_rec(root, EG);
    }
    static ADNodePtr extract_expr(int root_ec, const EGraph& EG, ADGraph& G) {
        ECostModel cm;
        std::unordered_map<int, ADNodePtr> memo;
        return extract_rec(root_ec, EG, cm, memo, G);
    }
    // Extract a full subgraph and return the new root, replacing stale pointers
    // in the old root's children with their memoized (extracted) versions.
    static ADNodePtr extract_subgraph(const ADNodePtr& old_root,
                                      int root_ec, const EGraph& EG, ADGraph& G) {
        // First, extract the full subgraph into fresh nodes
        ECostModel cm;
        std::unordered_map<int, ADNodePtr> memo;
        ADNodePtr new_root = extract_rec(root_ec, EG, cm, memo, G);

        // Now replace all references from old_root to new_root in G
        if (old_root && new_root && old_root.get() != new_root.get()) {
            G.replaceNodeReferences(old_root, new_root);
        }
        return new_root;
    }
};

// Helper: quick float quantization for stable zero/one checks
static inline double qfp_(double x, double s = 1e12) {
    return std::nearbyint(x * s) / s;
}
static inline bool is_zero_(double x) { return qfp_(x) == 0.0; }
static inline bool is_one_(double x) { return qfp_(x) == 1.0; }

// NOTE: Real AC is hard; we fake it by flattening/sorting in the bridge.
// Rules here assume Add/Multiply nodes are already flattened (n-ary).

// ---- Rule applications mutate egraph by *adding* canonical nodes and merging
// ----

static bool rule_add_zero_(EGraph& G, int ec) {
    // If Add contains zeros, drop them; if only one child remains, merge to it.
    bool changed = false;
    // Scan each representative node of this class
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Add || n.kids.size() < 2) continue;
        std::vector<EClassId> keep;
        for (auto k : n.kids) {
            int kc = G.find(k.id);
            if (G.classes[kc].has_const && is_zero_(G.classes[kc].cval)) {
                changed = true;
                continue;
            }
            keep.push_back({kc});
        }
        if (!changed) continue;
        if (keep.empty()) {
            // sum of zeros -> 0
            ENode c;
            c.op = Operator::cte;
            c.is_const = true;
            c.cval = 0.0;
            int z = G.add(c);
            G.merge(rep, z);
        } else if (keep.size() == 1) {
            G.merge(rep, keep[0].id);
        } else {
            ENode m;
            m.op = Operator::Add;
            m.kids = keep;
            std::sort(m.kids.begin(), m.kids.end(),
                      [](auto a, auto b) { return a.id < b.id; });
            int e2 = G.add(m);
            G.merge(rep, e2);
        }
        break;  // one application per rebuild is enough
    }
    return changed;
}

static bool rule_mul_one_zero_(EGraph& G, int ec) {
    bool changed = false;
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Multiply || n.kids.empty()) continue;

        bool has_zero = false;
        std::vector<EClassId> keep;
        for (auto k : n.kids) {
            int kc = G.find(k.id);
            if (G.classes[kc].has_const) {
                if (is_zero_(G.classes[kc].cval)) {
                    has_zero = true;
                    break;
                }
                if (is_one_(G.classes[kc].cval)) {
                    changed = true;
                    continue;
                }  // drop 1
            }
            keep.push_back({kc});
        }
        if (has_zero) {
            ENode c;
            c.op = Operator::cte;
            c.is_const = true;
            c.cval = 0.0;
            G.merge(rep, G.add(c));
            return true;
        }
        if (!changed) continue;
        if (keep.empty()) {
            ENode c;
            c.op = Operator::cte;
            c.is_const = true;
            c.cval = 1.0;
            G.merge(rep, G.add(c));
        } else if (keep.size() == 1) {
            G.merge(rep, keep[0].id);
        } else {
            ENode m;
            m.op = Operator::Multiply;
            m.kids = keep;
            std::sort(m.kids.begin(), m.kids.end(),
                      [](auto a, auto b) { return a.id < b.id; });
            G.merge(rep, G.add(m));
        }
        return true;
    }
    return false;
}

static bool rule_const_fold_add_mul_(EGraph& G, int ec) {
    bool changed = false;
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Add && n.op != Operator::Multiply) continue;
        double acc = (n.op == Operator::Add) ? 0.0 : 1.0;
        bool all_const = !n.kids.empty();
        for (auto k : n.kids) {
            int kc = G.find(k.id);
            if (!(G.classes[kc].has_const)) {
                all_const = false;
                break;
            }
            double v = G.classes[kc].cval;
            acc = (n.op == Operator::Add) ? acc + v : acc * v;
        }
        if (all_const) {
            ENode c;
            c.op = Operator::cte;
            c.is_const = true;
            c.cval = acc;
            G.merge(rep, G.add(c));
            changed = true;
            break;
        }
    }
    return changed;
}

static bool rule_distribute_left_(EGraph& G, int ec,
                                  const EGraphBudget& budget) {
    // x * (y1 + ... + yk) -> x*y1 + ... + x*yk, k <= max_distribute_terms
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Multiply || n.kids.size() != 2) continue;
        int L = G.find(n.kids[0].id);
        int R = G.find(n.kids[1].id);
        // Check if right is Add
        bool right_is_add = false;
        std::vector<EClassId> summands;
        for (int ridx : G.classes[R].nodes) {
            const ENode& rn = G.arena[ridx];
            if (rn.op == Operator::Add && rn.kids.size() >= 2) {
                right_is_add = true;
                for (auto s : rn.kids) summands.push_back({G.find(s.id)});
                break;
            }
        }
        if (!right_is_add || (int)summands.size() > budget.max_distribute_terms)
            continue;
        // Build x*yi
        std::vector<EClassId> terms;
        terms.reserve(summands.size());
        for (auto s : summands) {
            ENode mul;
            mul.op = Operator::Multiply;
            mul.kids = {{L}, {s.id}};
            if (mul.kids[0].id > mul.kids[1].id)
                std::swap(mul.kids[0], mul.kids[1]);
            int m = G.add(mul);
            terms.push_back({m});
        }
        // Sum them
        ENode add;
        add.op = Operator::Add;
        add.kids = terms;
        std::sort(add.kids.begin(), add.kids.end(),
                  [](auto a, auto b) { return a.id < b.id; });
        int sumc = G.add(add);
        G.merge(rep, sumc);
        return true;
    }
    return false;
}

static bool rule_factor_common_(EGraph& G, int ec) {
    // x*y1 + x*y2 + ... -> x*(y1 + y2 + ...)
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Add || n.kids.size() < 2) continue;

        // Collect a candidate common factor from first summand if it’s a
        // Multiply
        int first = G.find(n.kids[0].id);
        int cand = -1;
        for (int nidx : G.classes[first].nodes) {
            const ENode& mn = G.arena[nidx];
            if (mn.op == Operator::Multiply && mn.kids.size() >= 2) {
                cand = G.find(mn.kids[0].id);  // pick first factor as candidate
                break;
            }
        }
        if (cand < 0) continue;

        // Verify all summands contain cand as a direct factor
        std::vector<EClassId> rest_terms;
        bool ok = true;
        for (auto s : n.kids) {
            int sc = G.find(s.id);
            bool found = false;
            for (int sidx : G.classes[sc].nodes) {
                const ENode& sn = G.arena[sidx];
                if (sn.op == Operator::Multiply) {
                    // split: cand * other or other * cand
                    std::vector<EClassId> others;
                    for (auto f : sn.kids) {
                        if (G.find(f.id) == cand) {
                            found = true;
                        } else
                            others.push_back({G.find(f.id)});
                    }
                    if (found) {
                        if (others.empty()) {
                            // term is exactly cand → contributes 1
                            ENode c;
                            c.op = Operator::cte;
                            c.is_const = true;
                            c.cval = 1.0;
                            rest_terms.push_back({G.add(c)});
                        } else if (others.size() == 1) {
                            rest_terms.push_back({others[0].id});
                        } else {
                            ENode mul;
                            mul.op = Operator::Multiply;
                            mul.kids = others;
                            std::sort(
                                mul.kids.begin(), mul.kids.end(),
                                [](auto a, auto b) { return a.id < b.id; });
                            rest_terms.push_back({G.add(mul)});
                        }
                        break;
                    }
                }
            }
            if (!found) {
                ok = false;
                break;
            }
        }
        if (!ok) continue;

        // Build (sum of rest) then multiply by cand
        ENode sum;
        sum.op = Operator::Add;
        sum.kids = rest_terms;
        std::sort(sum.kids.begin(), sum.kids.end(),
                  [](auto a, auto b) { return a.id < b.id; });
        int sumc = G.add(sum);

        ENode mul;
        mul.op = Operator::Multiply;
        mul.kids = {{cand}, {sumc}};
        if (mul.kids[0].id > mul.kids[1].id)
            std::swap(mul.kids[0], mul.kids[1]);
        int fac = G.add(mul);

        G.merge(rep, fac);
        return true;
    }
    return false;
}

// ========== Inverse-composition rules ==========
// These rules only apply to nodes that are NOT already merged with
// a simpler representative. We guard against cycling by checking
// that the pattern's outer operator class has NOT been merged with
// the inner's child class already (i.e., rep != child_rep).

// log(exp(x)) → x (always valid since exp(x) > 0)
static bool rule_log_exp_(EGraph& G, int ec) {
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Log || n.kids.size() != 1) continue;
        // Get exp's child class (x's class)
        int inner_ec = G.find(n.kids[0].id); // exp class
        // Find exp's child
        int child_rep = -1;
        for (int sidx : G.classes[inner_ec].nodes) {
            const ENode& sn = G.arena[sidx];
            if (sn.op == Operator::Exp && sn.kids.size() == 1) {
                child_rep = G.find(sn.kids[0].id);
                break;
            }
        }
        if (child_rep < 0) continue;
        // Only fire if log class is NOT already merged with child
        if (G.find(rep) == child_rep) continue;
        G.merge(rep, child_rep);
        return true;
    }
    return false;
}

// exp(log(x)) → x (valid when x > 0)
static bool rule_exp_log_(EGraph& G, int ec) {
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Exp || n.kids.size() != 1) continue;
        int inner_ec = G.find(n.kids[0].id); // log class
        int child_rep = -1;
        for (int sidx : G.classes[inner_ec].nodes) {
            const ENode& sn = G.arena[sidx];
            if (sn.op == Operator::Log && sn.kids.size() == 1) {
                child_rep = G.find(sn.kids[0].id);
                break;
            }
        }
        if (child_rep < 0) continue;
        if (G.find(rep) == child_rep) continue;
        G.merge(rep, child_rep);
        return true;
    }
    return false;
}

// sqrt(x^2) → x (only when x is provably non-negative, e.g. exp(y))
static bool rule_sqrt_square_(EGraph& G, int ec) {
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Sqrt || n.kids.size() != 1) continue;
        int inner_ec = G.find(n.kids[0].id);
        // Check if inner is x^2 (i.e., Multiply(x, x))
        for (int sidx : G.classes[inner_ec].nodes) {
            const ENode& sn = G.arena[sidx];
            if (sn.op != Operator::Multiply || sn.kids.size() != 2) continue;
            int k0 = G.find(sn.kids[0].id);
            int k1 = G.find(sn.kids[1].id);
            if (k0 != k1) continue; // not x*x

            // Only apply if x is provably non-negative
            // Check: is x = exp(y)?  exp is always positive.
            bool nonneg = false;
            for (int xidx : G.classes[k0].nodes) {
                const ENode& xn = G.arena[xidx];
                if (xn.op == Operator::Exp) { nonneg = true; break; }
            }
            if (!nonneg) continue; // unsafe: x might be negative

            G.merge(rep, k0); // sqrt(x^2) → x
            return true;
        }
    }
    return false;
}

// sqrt(exp(x)) → exp(x/2)
static bool rule_sqrt_exp_(EGraph& G, int ec, const EGraphBudget&) {
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Sqrt || n.kids.size() != 1) continue;
        int inner_ec = G.find(n.kids[0].id);
        // Check if inner is exp(y)
        for (int sidx : G.classes[inner_ec].nodes) {
            const ENode& sn = G.arena[sidx];
            if (sn.op != Operator::Exp || sn.kids.size() != 1) continue;
            // sqrt(exp(y)) → exp(y/2)
            ENode cte_half;
            cte_half.op = Operator::cte;
            cte_half.is_const = true;
            cte_half.cval = 0.5;
            int cte_ec = G.add(cte_half);
            ENode half;
            half.op = Operator::Multiply;
            half.kids.emplace_back(sn.kids[0]);
            half.kids.emplace_back(EClassId{cte_ec});
            int half_ec = G.add(half);
            ENode exp_half;
            exp_half.op = Operator::Exp;
            exp_half.kids.emplace_back(EClassId{half_ec});
            int result_ec = G.add(exp_half);
            G.merge(rep, result_ec);
            return true;
        }
    }
    return false;
}

// abs(x) where x is provably non-negative → x
static bool rule_abs_nonneg_(EGraph& G, int ec) {
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Abs || n.kids.size() != 1) continue;
        int kc = G.find(n.kids[0].id);
        // Check if argument is provably non-negative (exp or constant >= 0)
        for (int sidx : G.classes[kc].nodes) {
            const ENode& sn = G.arena[sidx];
            if (sn.op == Operator::Exp) {
                int child_rep = G.find(sn.kids[0].id);
                if (G.find(rep) != child_rep) {
                    G.merge(rep, child_rep);
                }
                return true;
            }
            if (sn.is_const && sn.cval >= 0.0) {
                ENode c;
                c.op = Operator::cte;
                c.is_const = true;
                c.cval = sn.cval;
                G.merge(rep, G.add(c));
                return true;
            }
        }
    }
    return false;
}

// ========== Exp/Log composition rules ==========
// exp(a) * exp(b) → exp(a + b)  (valid for all real a, b)
static bool rule_exp_mul_add_(EGraph& G, int ec) {
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Multiply || n.kids.size() != 2) continue;
        int k0_ec = G.find(n.kids[0].id);
        int k1_ec = G.find(n.kids[1].id);
        // Find an exp in each child
        int a_ec = -1, b_ec = -1;
        for (int sidx : G.classes[k0_ec].nodes) {
            const ENode& sn = G.arena[sidx];
            if (sn.op == Operator::Exp && sn.kids.size() == 1) {
                a_ec = G.find(sn.kids[0].id); break;
            }
        }
        for (int sidx : G.classes[k1_ec].nodes) {
            const ENode& sn = G.arena[sidx];
            if (sn.op == Operator::Exp && sn.kids.size() == 1) {
                b_ec = G.find(sn.kids[0].id); break;
            }
        }
        if (a_ec < 0 || b_ec < 0) continue;
        // Build a + b
        ENode add_n;
        add_n.op = Operator::Add;
        add_n.kids = {{EClassId{a_ec}}, {EClassId{b_ec}}};
        std::sort(add_n.kids.begin(), add_n.kids.end(),
                  [](auto a, auto b) { return a.id < b.id; });
        int sum_ec = G.add(add_n);
        // Build exp(a + b)
        ENode exp_n;
        exp_n.op = Operator::Exp;
        exp_n.kids = {{EClassId{sum_ec}}};
        int result_ec = G.add(exp_n);
        G.merge(rep, result_ec);
        return true;
    }
    return false;
}

// exp(a) / exp(b) → exp(a - b)
static bool rule_exp_div_sub_(EGraph& G, int ec) {
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Divide || n.kids.size() != 2) continue;
        int num_ec = G.find(n.kids[0].id);
        int den_ec = G.find(n.kids[1].id);
        int a_ec = -1, b_ec = -1;
        for (int sidx : G.classes[num_ec].nodes) {
            const ENode& sn = G.arena[sidx];
            if (sn.op == Operator::Exp && sn.kids.size() == 1) {
                a_ec = G.find(sn.kids[0].id); break;
            }
        }
        for (int sidx : G.classes[den_ec].nodes) {
            const ENode& sn = G.arena[sidx];
            if (sn.op == Operator::Exp && sn.kids.size() == 1) {
                b_ec = G.find(sn.kids[0].id); break;
            }
        }
        if (a_ec < 0 || b_ec < 0) continue;
        // Build a - b
        ENode sub_n;
        sub_n.op = Operator::Subtract;
        sub_n.kids = {{EClassId{a_ec}}, {EClassId{b_ec}}};
        int diff_ec = G.add(sub_n);
        // Build exp(a - b)
        ENode exp_n;
        exp_n.op = Operator::Exp;
        exp_n.kids = {{EClassId{diff_ec}}};
        int result_ec = G.add(exp_n);
        G.merge(rep, result_ec);
        return true;
    }
    return false;
}

// log(a) + log(b) → log(a * b)  (valid when a, b > 0)
static bool rule_log_add_mul_(EGraph& G, int ec) {
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Add || n.kids.size() != 2) continue;
        int k0_ec = G.find(n.kids[0].id);
        int k1_ec = G.find(n.kids[1].id);
        int a_ec = -1, b_ec = -1;
        for (int sidx : G.classes[k0_ec].nodes) {
            const ENode& sn = G.arena[sidx];
            if (sn.op == Operator::Log && sn.kids.size() == 1) {
                a_ec = G.find(sn.kids[0].id); break;
            }
        }
        for (int sidx : G.classes[k1_ec].nodes) {
            const ENode& sn = G.arena[sidx];
            if (sn.op == Operator::Log && sn.kids.size() == 1) {
                b_ec = G.find(sn.kids[0].id); break;
            }
        }
        if (a_ec < 0 || b_ec < 0) continue;
        // Build a * b
        ENode mul_n;
        mul_n.op = Operator::Multiply;
        mul_n.kids = {{EClassId{a_ec}}, {EClassId{b_ec}}};
        std::sort(mul_n.kids.begin(), mul_n.kids.end(),
                  [](auto a, auto b) { return a.id < b.id; });
        int prod_ec = G.add(mul_n);
        // Build log(a * b)
        ENode log_n;
        log_n.op = Operator::Log;
        log_n.kids = {{EClassId{prod_ec}}};
        int result_ec = G.add(log_n);
        G.merge(rep, result_ec);
        return true;
    }
    return false;
}

// log(a / b) → log(a) - log(b)  (valid when a, b > 0)
static bool rule_log_div_sub_(EGraph& G, int ec) {
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Log || n.kids.size() != 1) continue;
        int div_ec = G.find(n.kids[0].id);
        // Check if inside is a Divide
        for (int sidx : G.classes[div_ec].nodes) {
            const ENode& sn = G.arena[sidx];
            if (sn.op != Operator::Divide || sn.kids.size() != 2) continue;
            int a_ec = G.find(sn.kids[0].id);
            int b_ec = G.find(sn.kids[1].id);
            // Build log(a) - log(b): two Log nodes, then Subtract
            ENode log_a_n;
            log_a_n.op = Operator::Log;
            log_a_n.kids = {{EClassId{a_ec}}};
            int log_a_ec = G.add(log_a_n);
            ENode log_b_n;
            log_b_n.op = Operator::Log;
            log_b_n.kids = {{EClassId{b_ec}}};
            int log_b_ec = G.add(log_b_n);
            ENode sub_n;
            sub_n.op = Operator::Subtract;
            sub_n.kids = {{EClassId{log_a_ec}}, {EClassId{log_b_ec}}};
            int result_ec = G.add(sub_n);
            G.merge(rep, result_ec);
            return true;
        }
    }
    return false;
}

// log(a^b) → b * log(a)  (valid for a > 0)
static bool rule_log_pow_(EGraph& G, int ec) {
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Log || n.kids.size() != 1) continue;
        int pow_ec = G.find(n.kids[0].id);
        for (int sidx : G.classes[pow_ec].nodes) {
            const ENode& sn = G.arena[sidx];
            if (sn.op != Operator::Pow || sn.kids.size() != 2) continue;
            int a_ec = G.find(sn.kids[0].id);  // base
            int b_ec = G.find(sn.kids[1].id);  // exponent
            // Build log(a)
            ENode log_n;
            log_n.op = Operator::Log;
            log_n.kids = {{EClassId{a_ec}}};
            int log_a_ec = G.add(log_n);
            // Build b * log(a)
            ENode mul_n;
            mul_n.op = Operator::Multiply;
            mul_n.kids = {{EClassId{b_ec}}, {EClassId{log_a_ec}}};
            std::sort(mul_n.kids.begin(), mul_n.kids.end(),
                      [](auto a, auto b) { return a.id < b.id; });
            int result_ec = G.add(mul_n);
            G.merge(rep, result_ec);
            return true;
        }
    }
    return false;
}

// ========== Negation / Sign rules ==========
// x / (-1) → -x
static bool rule_div_neg_one_(EGraph& G, int ec) {
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Divide || n.kids.size() != 2) continue;
        int num_ec = G.find(n.kids[0].id);
        int den_ec = G.find(n.kids[1].id);
        // Check if denominator is -1
        bool is_neg_one = false;
        for (int sidx : G.classes[den_ec].nodes) {
            const ENode& sn = G.arena[sidx];
            if (sn.op == Operator::Neg) {
                // Check if child is 1
                for (int cidx : G.classes[G.find(sn.kids[0].id)].nodes) {
                    const ENode& cn = G.arena[cidx];
                    if (cn.is_const && is_one_(cn.cval)) {
                        is_neg_one = true;
                        break;
                    }
                }
            }
            if (is_neg_one) break;
        }
        if (!is_neg_one) continue;
        // Build -x = Neg(x)
        ENode neg_n;
        neg_n.op = Operator::Neg;
        neg_n.kids = {{EClassId{num_ec}}};
        int neg_ec = G.add(neg_n);
        G.merge(rep, neg_ec);
        return true;
    }
    return false;
}

// (-1) * x → -x
static bool rule_mul_neg_one_(EGraph& G, int ec) {
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Multiply || n.kids.size() != 2) continue;
        // Check if one child is -1
        int x_ec = -1;
        for (int side = 0; side < 2; ++side) {
            int kid_ec = G.find(n.kids[side].id);
            bool is_neg_one = false;
            for (int sidx : G.classes[kid_ec].nodes) {
                const ENode& sn = G.arena[sidx];
                if (sn.op == Operator::Neg) {
                    for (int cidx : G.classes[G.find(sn.kids[0].id)].nodes) {
                        const ENode& cn = G.arena[cidx];
                        if (cn.is_const && is_one_(cn.cval)) {
                            is_neg_one = true;
                            break;
                        }
                    }
                }
                if (is_neg_one) break;
            }
            if (is_neg_one) {
                // The other child is x
                x_ec = G.find(n.kids[1 - side].id);
                break;
            }
        }
        if (x_ec < 0) continue;
        // Build -x
        ENode neg_n;
        neg_n.op = Operator::Neg;
        neg_n.kids = {{EClassId{x_ec}}};
        int neg_ec = G.add(neg_n);
        G.merge(rep, neg_ec);
        return true;
    }
    return false;
}

// -(-x) → x
static bool rule_neg_neg_(EGraph& G, int ec) {
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Neg || n.kids.size() != 1) continue;
        int inner_ec = G.find(n.kids[0].id);
        // Check if inner is also Neg
        for (int sidx : G.classes[inner_ec].nodes) {
            const ENode& sn = G.arena[sidx];
            if (sn.op == Operator::Neg && sn.kids.size() == 1) {
                int x_ec = G.find(sn.kids[0].id);
                G.merge(rep, x_ec);
                return true;
            }
        }
    }
    return false;
}

// -(a + b) → -(a) + -(b)
static bool rule_neg_add_(EGraph& G, int ec) {
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Neg || n.kids.size() != 1) continue;
        int add_ec = G.find(n.kids[0].id);
        // Check if inner is Add
        for (int sidx : G.classes[add_ec].nodes) {
            const ENode& sn = G.arena[sidx];
            if (sn.op != Operator::Add || sn.kids.size() != 2) continue;
            int a_ec = G.find(sn.kids[0].id);
            int b_ec = G.find(sn.kids[1].id);
            // Build -(a)
            ENode neg_a;
            neg_a.op = Operator::Neg;
            neg_a.kids = {{EClassId{a_ec}}};
            int neg_a_ec = G.add(neg_a);
            // Build -(b)
            ENode neg_b;
            neg_b.op = Operator::Neg;
            neg_b.kids = {{EClassId{b_ec}}};
            int neg_b_ec = G.add(neg_b);
            // Build -(a) + -(b)
            ENode add_n;
            add_n.op = Operator::Add;
            add_n.kids = {{EClassId{neg_a_ec}}, {EClassId{neg_b_ec}}};
            std::sort(add_n.kids.begin(), add_n.kids.end(),
                      [](auto a, auto b) { return a.id < b.id; });
            int result_ec = G.add(add_n);
            G.merge(rep, result_ec);
            return true;
        }
    }
    return false;
}

// 1 / x → x^(-1)
static bool rule_inv_to_pow_(EGraph& G, int ec) {
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Divide || n.kids.size() != 2) continue;
        // Check if numerator is 1
        int num_ec = G.find(n.kids[0].id);
        int x_ec = G.find(n.kids[1].id);
        bool num_is_one = false;
        for (int sidx : G.classes[num_ec].nodes) {
            const ENode& sn = G.arena[sidx];
            if (sn.is_const && is_one_(sn.cval)) {
                num_is_one = true;
                break;
            }
        }
        if (!num_is_one) continue;
        // Build x^(-1)
        ENode cte_neg_one;
        cte_neg_one.op = Operator::cte;
        cte_neg_one.is_const = true;
        cte_neg_one.cval = -1.0;
        int neg_one_ec = G.add(cte_neg_one);
        ENode pow_n;
        pow_n.op = Operator::Pow;
        pow_n.kids = {{EClassId{x_ec}}, {EClassId{neg_one_ec}}};
        int result_ec = G.add(pow_n);
        G.merge(rep, result_ec);
        return true;
    }
    return false;
}

// |exp(x)| → exp(x) (already handled in rule_abs_nonneg_)
// |x| = x when x >= 0 (already handled in rule_abs_nonneg_)
// Add: |x|^2 = x^2 (algebraic identity)
static bool rule_abs_sq_to_pow_(EGraph& G, int ec) {
    int rep = G.find(ec);
    for (int idx : G.classes[rep].nodes) {
        const ENode& n = G.arena[idx];
        if (n.op != Operator::Multiply || n.kids.size() != 2) continue;
        int k0_ec = G.find(n.kids[0].id);
        int k1_ec = G.find(n.kids[1].id);
        // Check if both children are |x| with the same x
        int x0_ec = -1, x1_ec = -1;
        for (int sidx : G.classes[k0_ec].nodes) {
            const ENode& sn = G.arena[sidx];
            if (sn.op == Operator::Abs && sn.kids.size() == 1) {
                x0_ec = G.find(sn.kids[0].id); break;
            }
        }
        for (int sidx : G.classes[k1_ec].nodes) {
            const ENode& sn = G.arena[sidx];
            if (sn.op == Operator::Abs && sn.kids.size() == 1) {
                x1_ec = G.find(sn.kids[0].id); break;
            }
        }
        if (x0_ec < 0 || x1_ec < 0 || x0_ec != x1_ec) continue;
        // |x| * |x| = x * x
        ENode mul_n;
        mul_n.op = Operator::Multiply;
        mul_n.kids = {{EClassId{x0_ec}}, {EClassId{x1_ec}}};
        std::sort(mul_n.kids.begin(), mul_n.kids.end(),
                  [](auto a, auto b) { return a.id < b.id; });
        int result_ec = G.add(mul_n);
        G.merge(rep, result_ec);
        return true;
    }
    return false;
}

// One saturation round over all classes
inline bool apply_rules_once(EGraph& G, const EGraphBudget& budget) {
    bool changed = false;
    for (int ec = 0; ec < (int)G.classes.size(); ++ec) {
        if (G.classes[ec].nodes.empty()) continue;
        changed |= rule_add_zero_(G, ec);
        changed |= rule_mul_one_zero_(G, ec);
        changed |= rule_const_fold_add_mul_(G, ec);
        if (changed) continue;  // rebuild soon
        changed |= rule_distribute_left_(G, ec, budget);
        if (changed) continue;
        changed |= rule_factor_common_(G, ec);
        if (changed) continue;
        // Inverse composition rules
        changed |= rule_log_exp_(G, ec);
        if (changed) continue;
        changed |= rule_exp_log_(G, ec);
        if (changed) continue;
        changed |= rule_sqrt_square_(G, ec);
        if (changed) continue;
        changed |= rule_sqrt_exp_(G, ec, budget);
        if (changed) continue;
        changed |= rule_abs_nonneg_(G, ec);
        if (changed) continue;
        // Exp/Log composition rules
        changed |= rule_exp_mul_add_(G, ec);
        if (changed) continue;
        changed |= rule_exp_div_sub_(G, ec);
        if (changed) continue;
        changed |= rule_log_add_mul_(G, ec);
        if (changed) continue;
        changed |= rule_log_div_sub_(G, ec);
        if (changed) continue;
        changed |= rule_log_pow_(G, ec);
        if (changed) continue;
        // Negation rules
        changed |= rule_div_neg_one_(G, ec);
        if (changed) continue;
        changed |= rule_mul_neg_one_(G, ec);
        if (changed) continue;
        changed |= rule_neg_neg_(G, ec);
        if (changed) continue;
        changed |= rule_neg_add_(G, ec);
        if (changed) continue;
        changed |= rule_inv_to_pow_(G, ec);
        if (changed) continue;
        changed |= rule_abs_sq_to_pow_(G, ec);
        if (changed) continue;
    }
    return changed;
}

// Driver: bounded saturation
inline void saturate(EGraph& G, const EGraphBudget& budget) {
    int it = 0;
    for (; it < budget.max_iterations; ++it) {
        bool any = apply_rules_once(G, budget);
        G.rebuild();
        if (!any) break;
        if (G.arena.size() > budget.max_nodes) break;
    }
}

// Debug: print egraph state
