#include "helpers.h"

#include <string>
#include <sstream>
#include <iterator>

bool read_features(std::istream& stream, Predictor::features& features) {
    std::string line;
    std::getline(stream, line);

    features.clear();
    std::istringstream linestream{line};
    double value;
    while (!linestream.eof()) {
        linestream >> value;
        features.push_back(value);
    }
    return !stream.eof();
}

std::vector<double> read_vector(std::istream& stream) {
    std::vector<double> result;

    std::copy(std::istream_iterator<double>(stream),
              std::istream_iterator<double>(),
              std::back_inserter(result));
    return result;
}
