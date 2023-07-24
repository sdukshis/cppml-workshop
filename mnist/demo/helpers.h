#include <istream>
#include <string>

#include <Eigen/Dense>

#include <mnist/classifier.h>

namespace mnist{

Eigen::MatrixXf read_mat_from_stream(size_t rows, size_t cols, std::istream& );

Eigen::MatrixXf read_mat_from_file(size_t rows, size_t cols, const std::string&);

bool read_features(std::istream& stream, mnist::Classifier::features_t& features);

std::vector<float> read_vector(std::istream&);

}