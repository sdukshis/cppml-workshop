#include <fstream>

#include <gtest/gtest.h>

#include <mnist/tf_classifier.h>

#include <helpers.h>

using namespace mnist;

const size_t width = 28;
const size_t height = 28;
const size_t output_dim = 10;


TEST(DISABLED_TfClassifier, predict_proba) {
    auto clf = TfClassifier{"train/saved_model", width, height};

    auto proba_true = TfClassifier::probas_t{};
    auto features = TfClassifier::features_t{};


    std::ifstream test_data{"train/test_mnist_cnn.txt"};
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

TEST(DISABLED_TfClassifier, predict_class) {
     auto clf = TfClassifier{"train/saved_model", width, height};

    auto features = TfClassifier::features_t{};


    std::ifstream test_data{"train/test_mnist_cnn_classes.txt"};
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