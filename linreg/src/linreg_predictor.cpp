#include <linreg_predictor.h>

#include <numeric>
#include <cassert>
#include <stdexcept>


LinregPredictor::LinregPredictor(const std::vector<double>& coef)
    : coef_{coef.at(0), coef.at(1)}
{
}

double LinregPredictor::predict(const features& feat) const {
    if (feat.size() == 0) {
        throw std::invalid_argument{"Incorrect feature vector size"};
    }
    return  coef_[0] + coef_[1]*feat[0];
}
