#pragma once

#include <string>
#include <memory>

#include <tensorflow/c/c_api.h>

#include "classifier.h"


namespace mnist {



class TfClassifier: public Classifier {
public:
    TfClassifier(const std::string& modelpath,
                 const int width,
                 const int height);

    TfClassifier(const TfClassifier&) = delete;

    TfClassifier& operator=(const TfClassifier&) = delete;

    size_t num_classes() const override;

    size_t predict(const features_t&) const override;

    probas_t predict_proba(const features_t&) const override;

protected:
    static void delete_tf_session(TF_Session*);

    using tf_graph = std::unique_ptr<TF_Graph, decltype(&TF_DeleteGraph)>;
    using tf_buffer = std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)>;
    using tf_import_graph_def_options = std::unique_ptr<TF_ImportGraphDefOptions, decltype(&TF_DeleteImportGraphDefOptions)>;
    using tf_status = std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)>;
    using tf_session_options = std::unique_ptr<TF_SessionOptions, decltype(&TF_DeleteSessionOptions)>;
    using tf_tensor = std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)>;
    using tf_session = std::unique_ptr<TF_Session, decltype(&delete_tf_session)>;

protected:
    tf_graph graph_{TF_NewGraph(), TF_DeleteGraph};
    tf_session_options session_opts_{TF_NewSessionOptions(), TF_DeleteSessionOptions};
    tf_session session_ = {nullptr, delete_tf_session};
    TF_Operation* input_op_ = nullptr;
    TF_Operation* output_op_ = nullptr;
    int width_;
    int height_;
};

}