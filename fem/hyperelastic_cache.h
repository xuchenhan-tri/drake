#pragma once

#include "drake/common/eigen_types.h"

namespace drake {
namespace fem {
    template <typename T>
    struct HyperelasticCache{};

    template <typename T>
    struct CorotatedLinearCache : public HyperelasticCache<T> {
        Matrix3<T> R;       // Corotation matrix.
        Matrix3<T> F;       // Deformation gradient.
        Matrix3<T> strain;  // Corotated Linear strain.
        T trace_strain;     // Trace of strain_.
    };
}
}
