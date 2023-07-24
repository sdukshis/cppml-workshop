#pragma once

#include "classifier.h"

#include <vector>

namespace kdd99 {

class LogregClassifier: public BinaryClassifier {
public:
    using coef_t = features_t;

    LogregClassifier(const coef_t& coef);

    float predict_proba(const features_t& feat) const override;

protected:
    std::vector<float> coef_;
};


}