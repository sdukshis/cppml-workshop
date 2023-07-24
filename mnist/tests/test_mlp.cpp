#include <fstream>

#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <mnist/mlp_classifier.h>

#include <helpers.h>

using namespace mnist;

const size_t input_dim = 784;
const size_t hidden_dim = 128;
const size_t output_dim = 10;

TEST(DISABLED_MlpClassifier, predict_proba) {
    auto w1 = read_mat_from_file(input_dim, hidden_dim, "train/w1.txt");
    auto w2 = read_mat_from_file(hidden_dim, output_dim, "train/w2.txt");

    auto clf = MlpClassifier{w1.transpose(), w2.transpose()};

    auto proba_true = MlpClassifier::probas_t{};
    auto features = MlpClassifier::features_t{};


    std::ifstream test_data{"train/test_mnist_mlp.txt"};
    ASSERT_TRUE(test_data.is_open());
    for (;;) {
        proba_true.clear();
        for (size_t i = 0; i < output_dim; ++i) {
            float val;
            test_data >> val;
            proba_true.push_back(val);
        }
        if (!read_features(test_data, features)) {
            break;
        }
        auto proba_pred = clf.predict_proba(features);
        ASSERT_EQ(proba_true.size(), proba_pred.size());
        for (size_t i = 0; i < output_dim; ++i) {
            ASSERT_NEAR(proba_true[i], proba_pred[i], 1e-5);
        }
    }
}

TEST(DISABLED_MlpClassifier, predict_class) {
    auto w1 = read_mat_from_file(input_dim, hidden_dim, "train/w1.txt");
    auto w2 = read_mat_from_file(hidden_dim, output_dim, "train/w2.txt");

    auto clf = MlpClassifier{w1.transpose(), w2.transpose()};

    auto features = MlpClassifier::features_t{};


    std::ifstream test_data{"train/test_mnist_mlp_classes.txt"};
    ASSERT_TRUE(test_data.is_open());
    for (;;) {
        size_t y_true;
        test_data >> y_true;
        if (!read_features(test_data, features)) {
            break;
        }
        auto y_pred = clf.predict(features);
        ASSERT_EQ(y_true, y_pred);
    }
}
