#pragma once

#include <istream>
#include <vector>

#include <predictor.h>

bool read_features(std::istream& stream, Predictor::features& features);

std::vector<double> read_vector(std::istream&);