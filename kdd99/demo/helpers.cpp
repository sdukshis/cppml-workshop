#include "helpers.h"

#include <string>
#include <sstream>
#include <iterator>

bool read_features(std::istream& stream, kdd99::BinaryClassifier::features_t& features) {
    std::string line;
    std::getline(stream, line);

    features.clear();
    std::istringstream linestream{line};
    double value;
    while (linestream, linestream >> value) {
        features.push_back(value);
    }
    return stream.good();
}

std::vector<float> read_vector(std::istream& stream) {
    std::vector<float> result;

    std::copy(std::istream_iterator<float>(stream),
              std::istream_iterator<float>(),
              std::back_inserter(result));
    return result;
}
