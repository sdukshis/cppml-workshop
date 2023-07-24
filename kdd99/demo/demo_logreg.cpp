#include <iostream>
#include <fstream>

#include <kdd99/logreg_classifier.h>

#include "helpers.h"

using namespace kdd99;
using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <parameters file>" << endl;
        return EXIT_FAILURE;
    }

    std::ifstream istream{argv[1]};
    auto coef = read_vector(istream);
    istream.close();

    auto predictor = LogregClassifier{coef};

    auto features = LogregClassifier::features_t{};

    while (read_features(cin, features)) {
        auto y_pred = predictor.predict_proba(features);
        cout << y_pred << "\n";
    };
}