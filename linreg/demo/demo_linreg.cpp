#include <iostream>
#include <fstream>

#include <linreg_predictor.h>

#include "helpers.h"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <parameters file>" << endl;
        return EXIT_FAILURE;
    }

    std::ifstream istream{argv[1]};
    auto coef = read_vector(istream);
    istream.close();

    auto predictor = LinregPredictor{coef};

    auto features = Predictor::features{};

    while (read_features(cin, features)) {
        auto y_pred = predictor.predict(features);
        cout << y_pred << "\n";
    };
}