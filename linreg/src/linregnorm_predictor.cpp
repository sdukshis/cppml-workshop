#include <linregnorm_predictor.h>
#include <stdexcept>

LinregnormPredictor::LinregnormPredictor(const std::vector<double> &coef,
                                         const std::vector<double> &mean,
                                         const std::vector<double> &std)
        : LinregPredictor{coef}
    {
    // Add your code here
    }
    
double LinregnormPredictor::predict(const features &feat) const {
    // Add your code here
    return 0.0;
}
