#pragma once

#include <string>
#include <memory>
#include "classifier.h"

#include <catboost/c_api.h>

namespace kdd99 {

class CatboostClassifier: public BinaryClassifier {
public:
    CatboostClassifier(const std::string& modepath);

    ~CatboostClassifier() override = default;
    
    float predict_proba(const features_t&) const override;

private:
    std::unique_ptr<ModelCalcerHandle, decltype(&ModelCalcerDelete)> model_; 
};

}