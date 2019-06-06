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
        out_dims.push_back(c->MakeDim(c->Dim(c->input(0), 2)));
        tensorflow::shape_inference::ShapeHandle out_shape = c->MakeShape(out_dims);
        c->set_output(0, out_shape);
//
//    return Status::OK();
      return Status::OK();
    });

void AddOneKernelLauncher(const float* in, const int N, float* out);

class AddOneOp : public OpKernel {
 public:
  explicit AddOneOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template flat<float>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    // Call the cuda kernel launcher
    AddOneKernelLauncher(input.data(), N, output.data());

  }
};

REGISTER_KERNEL_BUILDER(Name("AddOne").Device(DEVICE_GPU), AddOneOp);