#include <linreg_predictor.h>

#include <numeric>
#include <cassert>
#include <stdexcept>


LinregPredictor::LinregPredictor(const std::vector<double>& coef)
    : coef_{coef[0], coef[1]}
{
}

double LinregPredictor::predict(const features& feat) const {
    return  coef_[0] + coef_[1]*feat[0];
}
