#include <mean_predictor.h>

MeanPredictor::MeanPredictor(double mean)
    : mean_{mean}
{}

double MeanPredictor::predict(const features&) const {
    return mean_;
}
