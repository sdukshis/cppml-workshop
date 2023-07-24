#pragma once

#include "linreg_predictor.h"

class PolyPredictor: public LinregPredictor {
public:
    using LinregPredictor::LinregPredictor;
    
    double predict(const features&) const override;
};
