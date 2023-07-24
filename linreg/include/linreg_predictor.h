#pragma once

#include <array>

#include "predictor.h"

class LinregPredictor: public Predictor {
public:
    LinregPredictor(const std::vector<double>&);

    double predict(const features&) const override;
    
protected:
    std::array<double, 2> coef_;
};
