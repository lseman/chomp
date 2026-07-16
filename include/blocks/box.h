#pragma once

#include <Eigen/Core>

#include <optional>
#include <stdexcept>

#include <core/definitions.h>

namespace blocks {

[[nodiscard]] inline dvec clip_box(const dvec& x, const std::optional<dvec>& lb,
                                   const std::optional<dvec>& ub) {
    dvec y = x;
    if (lb) {
        if (lb->size() != x.size())
            throw std::invalid_argument("clip_box: lb size mismatch");
        y = y.cwiseMax(*lb);
    }
    if (ub) {
        if (ub->size() != x.size())
            throw std::invalid_argument("clip_box: ub size mismatch");
        y = y.cwiseMin(*ub);
    }
    return y;
}

}  // namespace blocks
