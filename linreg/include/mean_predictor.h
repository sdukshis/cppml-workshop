#pragma once

#include "predictor.h"

class MeanPredictor: public Predictor {
public:
    MeanPredictor(double mean);

    double predict(const features&) const override;

protected:
    double mean_;
};
