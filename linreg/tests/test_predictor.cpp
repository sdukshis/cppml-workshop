#include <fstream>
#include <iostream>

#include <gtest/gtest.h>

#include <linreg_predictor.h>
#include <linregnorm_predictor.h>
#include <poly_predictor.h>
#include <helpers.h>

TEST(DISABLED_LinregPredictor, compare_to_python) {
    std::vector<double> coef = {0.0, 0.0};

    auto predictor = LinregPredictor{coef};

    auto features = LinregPredictor::features{};

    double y_pred_expected = 0.0;

    std::ifstream test_data{"train/test_data_linreg.csv"};
    ASSERT_TRUE(test_data.is_open());
    for (;;) {
        test_data >> y_pred_expected;
        if (!read_features(test_data, features)) {
            break;
        }
        auto y_pred = predictor.predict(features);
        EXPECT_NEAR(y_pred_expected, y_pred, 1e-5);
    }
}

TEST(DISABLED_LinregPredictor, multi_features) {
    std::ifstream istream{"train/linreg_multi_coef.txt"};
    auto coef = read_vector(istream);
    istream.close();

    auto predictor = LinregPredictor{coef};

    auto features = LinregPredictor::features{};

    double y_pred_expected = 0.0;

    std::ifstream test_data{"train/test_data_linreg_multi.csv"};
    ASSERT_TRUE(test_data.is_open());
    for (;;) {
        test_data >> y_pred_expected;
        if (!read_features(test_data, features)) {
            break;
        }
        auto y_pred = predictor.predict(features);
        EXPECT_NEAR(y_pred_expected, y_pred, 1e-5);
    }
}

TEST(DISABLED_LinregNormPredictor, norm_features) {
    std::ifstream coef_file("train/linreg_norm_coef.txt");
    auto coef = read_vector(coef_file);
    std::ifstream mean_file("train/linreg_norm_mean.txt");
    auto mean = read_vector(mean_file);
    std::ifstream std_file("train/linreg_norm_std.txt");
    auto std = read_vector(std_file);

    auto predictor = LinregnormPredictor{coef, mean, std};

    auto features = LinregnormPredictor::features{};

    double y_pred_expected = 0.0;

    std::ifstream test_data{"train/test_data_linreg_norm.csv"};
    ASSERT_TRUE(test_data.is_open());
    for (;;) {
        test_data >> y_pred_expected;
        if (!read_features(test_data, features)) {
            break;
        }
        auto y_pred = predictor.predict(features);
        EXPECT_NEAR(y_pred_expected, y_pred, 1e-4);
    }
}

TEST(DISABLED_PolyPredictor, compare_to_python) {
    std::ifstream istream{"train/polyreg_coef.txt"};
    auto coef = read_vector(istream);
    istream.close();

    auto predictor = PolyPredictor{coef};

    auto features = PolyPredictor::features{};

    double y_pred_expected = 0.0;

    std::ifstream test_data{"train/test_data_polyreg.csv"};
    ASSERT_TRUE(test_data.is_open());
    for (;;) {
        test_data >> y_pred_expected;
        if (!read_features(test_data, features)) {
            break;
        }
        auto y_pred = predictor.predict(features);
        EXPECT_NEAR(y_pred_expected, y_pred, 1e-4);
    }
}
