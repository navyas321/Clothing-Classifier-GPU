
#define TILE_WIDTH 24
#define BATCH_SIZE 1000
#define H 28
#define W 28
#define H_out 24
#define W_out 24
#define K 5
#define C 1
#define M 50
#define X_tile_width 28
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_ 
#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

// Declares constant memory
__constant__ float constK[1250];

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B) {

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    __shared__ float xShared[784];

    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    #define kc4d(i3,i2,i1,i0) constK[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

    /*
        Your code here!
    */
    const int batch = blockIdx.x;
    const int h0 = threadIdx.y;
    const int w0 = threadIdx.x;


    float acc = 0.0;
    for(int b = batch;b < B; b+= BATCH_SIZE){
      for (int i = h0; i < X_tile_width; i += TILE_WIDTH) {
          for (int j = w0; j < X_tile_width; j += TILE_WIDTH)
            xShared[i*28+j] = x4d(b, 0, i, j);
      }
      __syncthreads();

      for(int m = 0; m<M; m++){
        acc = 0.0;
        #pragma unroll
        for (int p = 0; p < 5; p++){
            #pragma unroll
            for (int q = 0; q < 5; q++){
                acc += xShared[28*(h0+p)+(w0+q)] * kc4d(m, 0, p, q), acc;
            }
        }
        y4d(b,m,h0,w0) = acc;
      }
      __syncthreads();
    }

    #undef y4d
    #undef x4d
    #undef kc4d
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template<>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {
    

    // Use mxnet's CHECK_EQ to do assertions.

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    // cudaStream_t s = y.stream_->stream_;
    //cudaStream_t s = y.stream_->stream_;
    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...

    const int B = x.shape_[0];
    const int H_grid = ceil(H_out/TILE_WIDTH);
    const int W_grid = ceil(W_out/TILE_WIDTH);
    dim3 gridDim(BATCH_SIZE,1,1);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    // Copies to constant memory
    cudaMemcpyToSymbol(constK, w.dptr_, 5000, 0, cudaMemcpyDeviceToDevice);

    forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}


/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    assert( 0 && "No forward implementation for other datatypes needed for ECE408");
}

}
}

#endif
