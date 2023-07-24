#pragma once

#include "linreg_predictor.h"
#include <vector>

class LinregnormPredictor: public LinregPredictor {
public:
    LinregnormPredictor(const std::vector<double> &coef,
                        const std::vector<double> &mean,
                        const std::vector<double> &std);
    
    double predict(const features &) const override;
// Add your code here
};
