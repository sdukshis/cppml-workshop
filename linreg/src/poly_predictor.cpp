#include <poly_predictor.h>

#include <cassert>
#include <iostream>

double PolyPredictor::predict(const features& feat) const {
    return LinregPredictor::predict(feat);    
}
