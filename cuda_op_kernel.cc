#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("AddOne")
    .Input("input: float32")
    .Output("output: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
//        auto in_shape = c->input(0);
//        c->set_output(0, c->input(0));
        tensorflow::shape_inference::ShapeHandle in_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &in_shape));
//
//        // Construct and set the output shape
//        DimensionHandle out_d0, out_d1, out_d2, out_d3;
        std::vector<tensorflow::shape_inference::DimensionHandle> out_dims;
        out_dims.push_back(c->MakeDim(c->Dim(c->input(0), 0)));
        out_dims.push_back(c->MakeDim(c->Dim(c->input(0), 1)));
        tensorflow::shape_inference::ShapeHandle out_shape = c->MakeShape(out_dims);
        c->set_output(0, out_shape);
//
//    return Status::OK();
      return Status::OK();
    });

//void AddOneKernelLauncher(const float* in, const int N, float* out);
void AddOneKernelLauncher2(const float* in, float* out, float* max_init, const int batch_size, const int seq_len, const int n_features);

class AddOneOp : public OpKernel {
 public:
  explicit AddOneOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    // Get inputs
    const Tensor& input = ctx->input(0);

    // Setup output shape
    const TensorShape& input_shape(input.shape());
    TensorShape output_shape(input.shape());
//    output_shape.InsertDim(0, input_shape.dim_size(0));
    output_shape.RemoveDim(2);

    // Allocate output tensor
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));




    // Get the Eigen tensors and pass them on the launcher
    auto input_tensor   = input.tensor<float, 3>();
    auto output_tensor  = output->tensor<float, 2>();



    const int64 batch_size  = input_tensor.dimension(0);
    const int64 seq_len     = input_tensor.dimension(2);
    const int64 n_features  = input_tensor.dimension(1);


    Tensor max_init;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,  TensorShape({batch_size, n_features}), &max_init));
    auto max_init_tensor = max_init.tensor<float, 2>();

    AddOneKernelLauncher2(input_tensor.data(), output_tensor.data(), max_init_tensor.data(), batch_size, seq_len, n_features);
//    launchCapsulePrediction(ctx->eigen_device(), input_tensor, weights_tensor,
//      output_tensor);



  }
};

REGISTER_KERNEL_BUILDER(Name("AddOne").Device(DEVICE_GPU), AddOneOp);