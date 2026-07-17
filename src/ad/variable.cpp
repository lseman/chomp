#include <ad/expression.h>
#include <ad/variable.h>

// static
std::atomic<int> Variable::nextOrder_{0};

// Implicit lift Variable -> Expression
Variable::operator ExpressionPtr() const {
    auto self = std::const_pointer_cast<Variable>(shared_from_this());
    // Expression(VariablePtr, coeff)
    return std::make_shared<Expression>(self, 1.0);
}

// Track parent expressions (store weak_ptr to avoid cycles)
void Variable::addParent(const ExpressionPtr& parent) {
    if (parent) parents_.push_back(parent);
}

// Return strong refs to live parents (filter expired)
std::vector<ExpressionPtr> Variable::getParents() const {
    std::vector<ExpressionPtr> out;
    out.reserve(parents_.size());
    for (const auto& w : parents_) {
        if (auto p = w.lock()) out.push_back(std::move(p));
    }
    return out;
}

// Variable * Variable -> Expression
ExpressionPtr Variable::operator*(const Variable& other) const {
    auto g = std::make_shared<ADGraph>();
    auto self = std::const_pointer_cast<Variable>(shared_from_this());
    auto lhs  = std::make_shared<Expression>(self, 1.0, g);

    auto otherPtr = std::const_pointer_cast<Variable>(other.shared_from_this());
    auto rhs      = std::make_shared<Expression>(otherPtr, 1.0, g);

    return (*lhs) * (*rhs);
}

// Variable * scalar -> Expression
ExpressionPtr Variable::operator*(double lhs) const {
    auto self = std::const_pointer_cast<Variable>(shared_from_this());
    auto g    = std::make_shared<ADGraph>();
    auto e    = std::make_shared<Expression>(self, 1.0, g);
    return (*e) * lhs;
}
