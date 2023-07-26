#include <ios>
#include <iterator>
#include <mnist/tf_classifier.h>

#include <sstream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <cassert>

using mnist::TfClassifier;

void TfClassifier::delete_tf_session(TF_Session *tf_session) {
    tf_status status{TF_NewStatus(), TF_DeleteStatus};
    TF_DeleteSession(tf_session, status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
        std::stringstream ss;
        ss << " Unable to delete TF_Session: " << TF_Message(status.get());
        throw std::runtime_error{ss.str()};

    }
}

static void delete_buffer(void* data, size_t size) {
    delete[] reinterpret_cast<char*>(data);
}

static void dummy_deleter(void* data, size_t length, void* arg) {

}

TfClassifier::TfClassifier(const std::string& modelpath,
                           const int width,
                           const int height)
    : width_{width}
    , height_{height} {
  
    tf_status status{TF_NewStatus(), TF_DeleteStatus};   
    tf_import_graph_def_options opts{TF_NewImportGraphDefOptions(), TF_DeleteImportGraphDefOptions};

    TF_Buffer* RunOpts = NULL;
    const char *tags = "serve";
    int ntags = 1;

    session_.reset(TF_LoadSessionFromSavedModel(session_opts_.get(), RunOpts, modelpath.c_str(), &tags, 1, graph_.get(), nullptr, status.get()));
    if (TF_GetCode(status.get()) != TF_OK) {
        std::stringstream ss;
        ss << " Unable to import graph from '" << modelpath << "': " << TF_Message(status.get());
        throw std::invalid_argument{ss.str()};
    }
    size_t pos = 0;
    input_op_ = TF_GraphOperationByName(graph_.get(), "serving_default_input");
    if (input_op_ == nullptr) {
        throw std::runtime_error{"Input not found"};
    }
 
    output_op_ = TF_GraphOperationByName(graph_.get(), "StatefulPartitionedCall");
    if (output_op_ == nullptr) {
        throw std::runtime_error{"Output not found"};
    }
}


size_t TfClassifier::num_classes() const {
    return 10;
}

size_t TfClassifier::predict(const features_t& feat) const {
    auto proba = predict_proba(feat);
    // Add your code here
    return 0;
}

TfClassifier::probas_t TfClassifier::predict_proba(const features_t& feat) const {
    assert(width_ * height_ == feat.size());

    // Preprocess input features
    features_t preproc_features;
    preproc_features.reserve(feat.size());
    // Divide each bytes by 255
    std::transform(feat.begin(), feat.end(), std::back_inserter(preproc_features),
                   [](float val) { return val / 255;});
    std::vector<TF_Output> inputs;
    std::vector<TF_Tensor*> input_values;

    TF_Output input_opout = {input_op_, 0};
    inputs.push_back(input_opout);

    // Create variables to store the size of the input and output variables
    const int num_bytes_in = width_ * height_ * sizeof(float);
    const int num_bytes_out = num_classes() * sizeof(float);

    // Set input dimensions - this should match the dimensionality of the input in
    // the loaded graph, in this case it's three dimensional.
    int64_t in_dims[] = {1, width_, height_, 1};
    int64_t out_dims[] = {1, static_cast<int64_t>(num_classes())};

    tf_tensor input{TF_NewTensor(TF_FLOAT, in_dims, 4, reinterpret_cast<void*>(preproc_features.data()), num_bytes_in, &dummy_deleter, 0),
                    TF_DeleteTensor};
    input_values.push_back(input.get());

    std::vector<TF_Output> outputs;
    TF_Output output_opout = {output_op_, 0};
    outputs.push_back(output_opout);

    // Create TF_Tensor* vector
    std::vector<TF_Tensor*> output_values(outputs.size(), nullptr);

    // Similar to creating the input tensor, however here we don't yet have the
    // output values, so we use TF_AllocateTensor()
    tf_tensor output_value{TF_AllocateTensor(TF_FLOAT, out_dims, 2, num_bytes_out), TF_DeleteTensor};
    output_values.push_back(output_value.get());

    tf_status status{TF_NewStatus(), TF_DeleteStatus};

    TF_SessionRun(session_.get(), nullptr,
                &inputs[0], &input_values[0], inputs.size(),
                &outputs[0], &output_values[0], outputs.size(),
                nullptr, 0, nullptr, status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
        std::stringstream ss;
        ss << "Unable to run session from graph: " << TF_Message(status.get());
        throw std::runtime_error{ss.str()};
    }

    probas_t probas;
    float* out_vals = static_cast<float*>(TF_TensorData(output_values[0]));
    for (size_t i = 0; i < num_classes(); ++i) {
        probas.push_back(*out_vals++);
    }

    return probas;
}