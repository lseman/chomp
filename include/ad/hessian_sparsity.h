#pragma once

#include "ad_graph.h"
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ad_detail {

// Structural pattern, independent of x and mutable Lagrange multipliers.
// Columns with no common nonzero row can share one H*v seed direction.
struct HessianSparsity {
    std::vector<std::vector<int>> rows;
    std::vector<int> color;
    int colors = 0;
    bool compressed = false;

    void build(const ADNodePtr& root, const std::vector<ADNodePtr>& variables) {
        const size_t n = variables.size();
        rows.assign(n, {});
        color.assign(n, -1);
        colors = 0;
        compressed = false;
        if (n < 32 || !root) return;
        const size_t budget = std::max<size_t>(4096, std::min<size_t>(n * 128, 1000000));
        size_t edges = 0, dependency_entries = 0;
        bool exceeded = false;
        std::unordered_map<const ADNode*, int> axis;
        for (size_t i = 0; i < n; ++i) axis.emplace(variables[i].get(), static_cast<int>(i));
        std::unordered_map<const ADNode*, std::vector<int>> dependencies;

        // Query only operands of nonlinear operations. In particular, don't
        // materialize the supports of every prefix of an outer linear sum.
        auto support = [&](const ADNode* operand) -> const std::vector<int>& {
            std::vector<std::pair<const ADNode*, bool>> stack{{operand, false}};
            while (!stack.empty() && !exceeded) {
                auto [node, expanded] = stack.back();
                stack.pop_back();
                if (dependencies.contains(node)) continue;
                if (auto it = axis.find(node); it != axis.end()) {
                    dependencies[node] = {it->second};
                    ++dependency_entries;
                } else if (!node || node->inputs.empty()) {
                    dependencies[node] = {};
                } else if (!expanded) {
                    stack.emplace_back(node, true);
                    for (const auto& child : node->inputs)
                        if (child && !dependencies.contains(child.get())) stack.emplace_back(child.get(), false);
                } else {
                    std::vector<int> merged;
                    for (const auto& child : node->inputs) {
                        if (!child) continue;
                        const auto& indices = dependencies.at(child.get());
                        merged.insert(merged.end(), indices.begin(), indices.end());
                    }
                    std::sort(merged.begin(), merged.end());
                    merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
                    dependency_entries += merged.size();
                    dependencies[node] = std::move(merged);
                }
                if (dependency_entries > budget) exceeded = true;
            }
            // An exceeded budget selects the existing uncompressed path.
            return dependencies[operand];
        };
        auto cross = [&](const std::vector<int>& a, const std::vector<int>& b) {
            if (exceeded) return;
            if (!a.empty() && b.size() > (budget - std::min(edges, budget)) / (2 * a.size())) {
                exceeded = true;
                return;
            }
            for (int i : a) for (int j : b) {
                rows[i].push_back(j);
                rows[j].push_back(i);
                edges += 2;
            }
        };

        std::unordered_set<const ADNode*> seen;
        std::vector<const ADNode*> stack{root.get()};
        while (!stack.empty() && !exceeded) {
            const ADNode* node = stack.back();
            stack.pop_back();
            if (!node || !seen.insert(node).second) continue;
            for (const auto& child : node->inputs) if (child) stack.push_back(child.get());
            switch (node->type) {
            case Operator::cte: case Operator::Var:
            case Operator::Add: case Operator::Subtract: case Operator::Neg:
            case Operator::Abs: case Operator::Relu: case Operator::Min: case Operator::Max:
                break; // locally linear; descendants still contribute curvature
            case Operator::Multiply: {
                std::vector<const ADNode*> operands;
                for (const auto& child : node->inputs)
                    if (child && child->type != Operator::cte &&
                        (!child->inputs.empty() || axis.contains(child.get()))) operands.push_back(child.get());
                for (size_t i = 0; i < operands.size() && !exceeded; ++i)
                    for (size_t j = i + 1; j < operands.size() && !exceeded; ++j) {
                        const auto& a = support(operands[i]);
                        const auto& b = support(operands[j]);
                        cross(a, b);
                    }
                break;
            }
            case Operator::Divide: {
                if (node->inputs.size() != 2) { exceeded = true; break; }
                const auto& a = support(node->inputs[0].get());
                const auto& b = support(node->inputs[1].get());
                cross(a, b);
                cross(b, b);
                break;
            }
            case Operator::Pow:
                if (node->inputs.size() == 2 && node->inputs[1]->type == Operator::cte &&
                    (node->inputs[1]->value == 0.0 || node->inputs[1]->value == 1.0)) break;
                [[fallthrough]];
            case Operator::Sin: case Operator::Cos: case Operator::Tan:
            case Operator::Exp: case Operator::Log: case Operator::Tanh:
            case Operator::Silu: case Operator::Gelu: case Operator::Softmax: case Operator::Sqrt: {
                std::vector<int> combined;
                for (const auto& child : node->inputs) {
                    const auto& indices = support(child.get());
                    combined.insert(combined.end(), indices.begin(), indices.end());
                }
                std::sort(combined.begin(), combined.end());
                combined.erase(std::unique(combined.begin(), combined.end()), combined.end());
                cross(combined, combined);
                break;
            }
            default: exceeded = true; break;
            }
        }
        if (exceeded) { rows.clear(); color.clear(); return; }
        for (auto& row : rows) {
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
        }
        std::vector<int> forbidden(n, -1);
        for (size_t col = 0; col < n; ++col) {
            if (rows[col].empty()) continue;
            for (int row : rows[col]) for (int other : rows[row])
                if (color[other] >= 0) forbidden[color[other]] = static_cast<int>(col);
            int next = 0;
            while (forbidden[next] == static_cast<int>(col)) ++next;
            color[col] = next;
            colors = std::max(colors, next + 1);
        }
        compressed = colors < static_cast<int>(n / 2);
        if (!compressed) { rows.clear(); color.clear(); }
    }
};

} // namespace ad_detail
