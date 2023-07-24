#pragma once

#include <vector>

namespace kdd99 {

class BinaryClassifier {
public:

    using features_t = std::vector<float>;

    virtual ~BinaryClassifier() {}

    virtual float predict_proba(const features_t&) const = 0;
};

}