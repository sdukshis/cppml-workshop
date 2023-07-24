#include <kdd99/logreg_classifier.h>

#include <cassert>
#include <numeric>
#include <cmath>
#include <stdexcept>

using kdd99::LogregClassifier;

namespace {

template<typename T>
auto sigma(T x) {
    return 1/(1 + std::exp(-x));
}

}

LogregClassifier::LogregClassifier(const coef_t& coef)
    : coef_{coef} 
{
    assert(!coef_.empty());
}

float LogregClassifier::predict_proba(const features_t& feat) const {
    if (feat.size() + 1 != coef_.size()) {
        throw std::runtime_error("LogregClassifier::predict_proba: "
                                 "feature vector size mismatch");
    }
    auto z = std::inner_product(feat.begin(), feat.end(), ++coef_.begin(), coef_.front());
    return sigma(z);
}
