#include <stdio.h>                                                                        
#include <stdlib.h>                                                                       
#include <tensorflow/c/c_api.h>                                                           

#include <vector>
#include <iostream>

TF_Buffer* read_file(const char* file);                                                   

void free_buffer(void* data, size_t length) {                                             
        free(data);                                                                       
}

void Deallocator(void* data, size_t length, void* arg) {
        std::cout << "Deallocator called\n";
        // free(data);
        // *reinterpret_cast<bool*>(arg) = true;
}


int main(int argc, char* argv[]) {
  if (argc < 2) {
      fprintf(stderr, "Usage: %s modelpath\n", argv[0]);
      return 1;
  }                                                                              
  // Graph definition from unzipped https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
  // which is used in the Go, Java and Android examples                                   
  TF_Buffer* graph_def = read_file(argv[1]);                      
  TF_Graph* graph = TF_NewGraph();

  // Import graph_def into graph                                                          
  TF_Status* status = TF_NewStatus();                                                     
  TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();                         
  TF_GraphImportGraphDef(graph, graph_def, opts, status);
  TF_DeleteImportGraphDefOptions(opts);
  if (TF_GetCode(status) != TF_OK) {
          fprintf(stderr, "ERROR: Unable to import graph %s", TF_Message(status));        
          return 1;
  }       
  fprintf(stdout, "Successfully imported graph\n");

  TF_SessionOptions* session_opts = TF_NewSessionOptions();
  TF_Session* session = TF_NewSession(graph, session_opts, status);
  if (TF_GetCode(status) != TF_OK) {
          fprintf(stderr, "ERROR: Unable to create session from graph %s", TF_Message(status));        
          return 1;
  } 
  fprintf(stdout, "Successfully created session\n");

  TF_Buffer* run_options = NULL;

 // Create variables to store the size of the input and output variables
  const int num_bytes_in = 28 * 28 * sizeof(float);
  const int num_bytes_out = 10 * sizeof(float);

  // Set input dimensions - this should match the dimensionality of the input in
  // the loaded graph, in this case it's three dimensional.
  int64_t in_dims[] = {1, 28, 28, 1};
  int64_t out_dims[] = {1, 10};

  size_t pos = 0;
  TF_Operation* oper;
  while ((oper = TF_GraphNextOperation(graph, &pos)) != nullptr) {
      std::cout << "Input: " << TF_OperationName(oper) << "\n";
  }
  // ######################
  // Set up graph inputs
  // ######################

  // Create a variable containing your values, in this case the input is a
  // 3-dimensional float
  float values[28*28] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.329412, 0.72549, 0.623529, 0.592157, 0.235294, 0.141176, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.870588, 0.996078, 0.996078, 0.996078, 0.996078, 0.945098, 0.776471, 0.776471, 0.776471, 0.776471, 0.776471, 0.776471, 0.776471, 0.776471, 0.666667, 0.203922, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.262745, 0.447059, 0.282353, 0.447059, 0.639216, 0.890196, 0.996078, 0.882353, 0.996078, 0.996078, 0.996078, 0.980392, 0.898039, 0.996078, 0.996078, 0.54902, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0666667, 0.258824, 0.054902, 0.262745, 0.262745, 0.262745, 0.231373, 0.0823529, 0.92549, 0.996078, 0.415686, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.32549, 0.992157, 0.819608, 0.0705882, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0862745, 0.913725, 1, 0.32549, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.505882, 0.996078, 0.933333, 0.172549, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.231373, 0.976471, 0.996078, 0.243137, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.521569, 0.996078, 0.733333, 0.0196078, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0352941, 0.803922, 0.972549, 0.227451, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.494118, 0.996078, 0.713726, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.294118, 0.984314, 0.941176, 0.223529, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0745098, 0.866667, 0.996078, 0.65098, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0117647, 0.796078, 0.996078, 0.858824, 0.137255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.14902, 0.996078, 0.996078, 0.301961, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.121569, 0.878431, 0.996078, 0.45098, 0.00392157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.521569, 0.996078, 0.996078, 0.203922, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.239216, 0.94902, 0.996078, 0.996078, 0.203922, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.47451, 0.996078, 0.996078, 0.858824, 0.156863, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.47451, 0.996078, 0.811765, 0.0705882, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  // Create vectors to store graph input operations and input tensors
  std::vector<TF_Output> inputs;
  std::vector<TF_Tensor*> input_values;

  // Pass the graph and a string name of your input operation
  // (make sure the operation name is correct)
  TF_Operation* input_op = TF_GraphOperationByName(graph, "conv2d_1_input");
  if (input_op == nullptr) {
    fprintf(stderr, "Operation 'conv2d_18' not found in graph\n");
    return 1;
  }
  std::cout << "Input op info: " << TF_OperationNumOutputs(input_op) << "\n";

  TF_Output input_opout = {input_op, 0};
  inputs.push_back(input_opout);

  // Create the input tensor using the dimension (in_dims) and size (num_bytes_in)
  // variables created earlier
  TF_Tensor* input = TF_NewTensor(TF_FLOAT, in_dims, 4, values, num_bytes_in, &Deallocator, 0);
  if (input == nullptr) {
      std::cerr << "Error: TF_NewTensor" << std::endl;
      return 1;
  }
  // Optionally, you can check that your input_op and input tensors are correct
  // by using some of the functions provided by the C API.
  std::cout << "Input data info: " << TF_Dim(input, 0) << "\n";

  input_values.push_back(input);


  // ######################
  // Set up graph outputs (similar to setting up graph inputs)
  // ######################

  // Create vector to store graph output operations
  std::vector<TF_Output> outputs;
  TF_Operation* output_op = TF_GraphOperationByName(graph, "dense_4/Softmax");
  if (output_op == nullptr) {
    fprintf(stderr, "Operation 'dense_4/Softmax' not found in graph\n");
    return 1;
  }
  TF_Output output_opout = {output_op, 0};
  outputs.push_back(output_opout);

  // Create TF_Tensor* vector
  std::vector<TF_Tensor*> output_values(outputs.size(), nullptr);

  // Similar to creating the input tensor, however here we don't yet have the
  // output values, so we use TF_AllocateTensor()
  TF_Tensor* output_value = TF_AllocateTensor(TF_FLOAT, out_dims, 2, num_bytes_out);
  output_values.push_back(output_value);

  // As with inputs, check the values for the output operation and output tensor
  std::cout << "Output: " << TF_OperationName(output_op) << "\n";
  std::cout << "Output info: " << TF_Dim(output_value, 0) << "\n";
  std::cout << "Output info: " << TF_Dim(output_value, 1) << "\n";


  // TF_Output* inputs = NULL;
  // TF_Tensor*  input_values = NULL;
  // int ninputs = 0;
  // TF_Output* outputs = NULL;
  // TF_Tensor* output_values = NULL;
  // int noutputs = 0;
  // // Target operations
  // TF_Operation* target_opers = NULL;
  // int ntargets = 0;
    
  // TF_Buffer* run_metadata = NULL;
 // Call TF_SessionRun
  TF_SessionRun(session, nullptr,
                &inputs[0], &input_values[0], inputs.size(),
                &outputs[0], &output_values[0], outputs.size(),
                nullptr, 0, nullptr, status);
  // TF_SessionRun(session, run_options, 
  //   inputs, &input_values, ninputs,
  //   outputs, &output_values, noutputs,
  //   &target_opers, ntargets,
  //   run_metadata,
  //   status
  // );
  if (TF_GetCode(status) != TF_OK) {
          fprintf(stderr, "ERROR: Unable to run session %s", TF_Message(status));        
          return 1;
  }
  float* out_vals = static_cast<float*>(TF_TensorData(output_values[0]));
  for (int i = 0; i < 10; ++i)
  {
      std::cout << "Output values info: " << *out_vals++ << "\n";
  }

  fprintf(stdout, "Successfully run session\n");

  TF_CloseSession(session, status);
  if (TF_GetCode(status) != TF_OK) {
          fprintf(stderr, "ERROR: Unable to close session %s", TF_Message(status));        
          return 1;
  }
  TF_DeleteSession(session, status);
  if (TF_GetCode(status) != TF_OK) {
          fprintf(stderr, "ERROR: Unable to delete session %s", TF_Message(status));        
          return 1;
  }   
  TF_DeleteSessionOptions(session_opts);                                     
  TF_DeleteStatus(status);
  TF_DeleteBuffer(graph_def);                                                             

  // Use the graph                                                                        
  TF_DeleteGraph(graph);                                                                  
  return 0;
} 

TF_Buffer* read_file(const char* file) {                                                  
  FILE *f = fopen(file, "rb");
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);                                                                  
  fseek(f, 0, SEEK_SET);  //same as rewind(f);                                            

  void* data = malloc(fsize);                                                             
  fread(data, fsize, 1, f);
  fclose(f);

  TF_Buffer* buf = TF_NewBuffer();                                                        
  buf->data = data;
  buf->length = fsize;                                                                    
  buf->data_deallocator = free_buffer;                                                    
  return buf;
} 