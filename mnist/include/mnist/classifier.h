#pragma once

#include <vector>
#include <cstddef>

namespace mnist {

class Classifier {
public:
    using features_t = std::vector<float>;
    using probas_t = std::vector<float>;

    virtual ~Classifier() {}

    virtual size_t num_classes() const = 0;

    virtual size_t predict(const features_t&) const = 0;

    virtual probas_t predict_proba(const features_t&) const = 0;
};


}