#pragma once

#include <vector>

class Predictor {
public:
    using features = std::vector<double>;

    virtual ~Predictor() {};
    
    virtual double predict(const features&) const = 0;
};
