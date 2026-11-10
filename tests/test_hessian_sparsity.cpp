#include <ad/hessian_sparsity.h>
#include <cassert>

static ADNodePtr node(Operator op, std::vector<ADNodePtr> inputs = {}) {
    auto result = std::make_shared<ADNode>();
    result->type = op;
    result->inputs = std::move(inputs);
    return result;
}

int main() {
    std::vector<ADNodePtr> x;
    for (int i = 0; i < 64; ++i) x.push_back(node(Operator::Var));
    auto two = node(Operator::cte);
    two->value = 2.0;
    ADNodePtr diagonal, banded;
    for (size_t i = 0; i < x.size(); ++i) {
        auto square = node(Operator::Pow, {x[i], two});
        diagonal = diagonal ? node(Operator::Add, {diagonal, square}) : square;
        if (i > 0) {
            auto difference = node(Operator::Subtract, {x[i], x[i - 1]});
            auto term = node(Operator::Pow, {difference, two});
            banded = banded ? node(Operator::Add, {banded, term}) : term;
        }
    }
    ad_detail::HessianSparsity pattern;
    pattern.build(diagonal, x);
    assert(pattern.compressed && pattern.colors == 1);
    for (size_t i = 0; i < x.size(); ++i) assert(pattern.rows[i] == std::vector<int>{static_cast<int>(i)});
    pattern.build(node(Operator::Multiply, {two, banded}), x);
    assert(pattern.compressed && pattern.colors == 3);
    for (const auto& row : pattern.rows) {
        std::unordered_set<int> colors;
        for (int col : row) assert(colors.insert(pattern.color[col]).second);
    }

    auto multiplier = node(Operator::Var); // mutable parameter, initially zero
    auto constraint = node(Operator::Multiply, {x.front(), x.back()});
    pattern.build(node(Operator::Add, {diagonal, node(Operator::Multiply, {multiplier, constraint})}), x);
    assert(pattern.compressed);
    assert(std::binary_search(pattern.rows.front().begin(), pattern.rows.front().end(), 63));

    ADNodePtr sum = x.front();
    for (size_t i = 1; i < x.size(); ++i) sum = node(Operator::Add, {sum, x[i]});
    pattern.build(node(Operator::Pow, {sum, two}), x);
    assert(!pattern.compressed); // dense structure selects original basis sweeps
    pattern.build(x.front(), x);
    assert(pattern.compressed && pattern.colors == 0);
}
