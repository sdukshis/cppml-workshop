#include <iostream>
#include <sstream>


#include <mean_predictor.h>

#include "helpers.h"

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " mean" << endl;
        return EXIT_FAILURE;
    }

    double mean = std::stod(argv[1]);

    auto predictor = MeanPredictor{mean};

    auto features = Predictor::features{};

    while (read_features(cin, features)) {
        for (auto f: features) {
            clog << f << ",";
        }
        clog << "\n";
        auto y_pred = predictor.predict(features);
        cout << y_pred << "\n";
    };
}