#pragma once

#include <istream>
#include <vector>

#include <kdd99/classifier.h>

bool read_features(std::istream& stream, kdd99::BinaryClassifier::features_t& features);

std::vector<float> read_vector(std::istream&);