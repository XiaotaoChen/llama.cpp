#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <string>
#include <fstream> // Add this line to include the necessary header
#include <stdexcept>
#include <sstream>  // stringstream
#include <vector>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>

#define MAX_NARGS 2

#define TN_MVLM_PROJ_MLP "mm.model.mlp.%d.%s"
#define TN_MVLM_PROJ_BLOCK "mm.model.mb_block.%d.block.%d.%s"
#define TN_MVLM_PROJ_PEG_MLP "mm.mlp.%d.%s"
#define TN_MVLM_PROJ_PEG "mm.peg.proj.%d.%s"


#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#endif

//
// logging
//
#define GGML_DEBUG 0
#if (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG(...)
#endif

#if (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_5(...)
#endif

#if (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_10(...)
#endif

#define GGML_PRINT(...) printf(__VA_ARGS__)

#include<vector>

static float frand(void) {
    return (float)rand()/(float)RAND_MAX;
}

static struct ggml_tensor * get_random_tensor(
    struct ggml_context * ctx0, int ndims, int64_t ne[], float fmin, float fmax
) {
    struct ggml_tensor * result = ggml_new_tensor(ctx0, GGML_TYPE_F32, ndims, ne);

    switch (ndims) {
        case 1:
            for (int i0 = 0; i0 < ne[0]; i0++) {
                ((float *)result->data)[i0] = frand()*(fmax - fmin) + fmin;
            }
            break;
        case 2:
            for (int i1 = 0; i1 < ne[1]; i1++) {
                for (int i0 = 0; i0 < ne[0]; i0++) {
                    ((float *)result->data)[i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                }
            }
            break;
        case 3:
            for (int i2 = 0; i2 < ne[2]; i2++) {
                for (int i1 = 0; i1 < ne[1]; i1++) {
                    for (int i0 = 0; i0 < ne[0]; i0++) {
                        ((float *)result->data)[i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                    }
                }
            }
            break;
        case 4:
            for (int i3 = 0; i3 < ne[3]; i3++) {
                for (int i2 = 0; i2 < ne[2]; i2++) {
                    for (int i1 = 0; i1 < ne[1]; i1++) {
                        for (int i0 = 0; i0 < ne[0]; i0++) {
                            ((float *)result->data)[i3*ne[2]*ne[1]*ne[0] + i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                        }
                    }
                }
            }
            break;
        default:
            assert(false);
    }

    return result;
}
static struct ggml_tensor * get_f16_tensor(
    struct ggml_context * ctx0, const ggml_tensor* f32_tensor) {
    struct ggml_tensor * result = ggml_new_tensor(ctx0, GGML_TYPE_F16, ggml_n_dims(f32_tensor), f32_tensor->ne);
    assert (ggml_nelements(f32_tensor) == ggml_nelements(result));
    for (int i = 0; i < ggml_nelements(f32_tensor); i++) {
        ((ggml_fp16_t *)result->data)[i] = ggml_fp32_to_fp16(((float *)f32_tensor->data)[i]);
    }
    return result;
}

static void printData(const float* data, int batch, int channel, int h, int w, const char name[]) {
    printf("========== %s ======== \n", name);
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channel; ++c) {
            for (int i = 0; i < h; ++i) {
                printf("[%d, %d, %d]: ", b, c, i);
                for (int j = 0; j < w; j++) {
                    printf("%.3f ", data[((b * channel + c) * h + i) * w + j]);
                }
                printf("\n");
            }
        }
    }
    printf("\n");
}


static void print_tensor_info(const ggml_tensor* tensor, const char* prefix = "") {
    size_t tensor_size = ggml_nbytes(tensor);
    printf("%s: n_dims = %d, name = %s, tensor_size=%zu, shape:[%d, %d, %d, %d], type: %d\n",
            prefix, ggml_n_dims(tensor), tensor->name, tensor_size,
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], tensor->type);
}

void reference_conv2d(const float* input, const float* weight,
                             const float* bias, std::vector<float>& output,
                             int batch, int ic, int oc, int ih, int iw, int pad_h, int pad_w,
                             int kh, int kw, int stride, int dilation, int group) {
    int oh, ow;
    oh = (ih + 2 * pad_h - (kh - 1) * dilation - 1) / stride + 1;
    ow = (iw + 2 * pad_w - (kw - 1) * dilation - 1) / stride + 1;

    assert(oc % group == 0 && ic % group == 0);
    output.resize(batch * oh * ow * oc);
    int oc_step = oc / group, ic_step = ic / group;
    for (int b = 0; b < batch; ++b) {
        for (int o_c = 0; o_c < oc; ++o_c) {
            for (int o_h = 0; o_h < oh; ++o_h) {
                for (int o_w = 0; o_w < ow; ++o_w) {
                    float result_data = 0;
                    int g = o_c / oc_step;
                    for (int i_c = g * ic_step; i_c < (g + 1) * ic_step; ++i_c) {
                        for (int k_h = 0; k_h < kh; ++k_h) {
                            for (int k_w = 0; k_w < kw; ++k_w) {
                                int i_h = o_h * stride - pad_h + k_h * dilation;
                                int i_w = o_w * stride - pad_w + k_w * dilation;
                                if (i_h < 0 || i_h >= ih || i_w < 0 || i_w >= iw) {
                                    continue;
                                }
                                float input_data = input[((b * ic + i_c) * ih + i_h) * iw + i_w];
                                float weight_data = weight[(((g * oc_step + o_c % oc_step) * ic_step + i_c % ic_step) * kh + k_h) * kw + k_w];
                                result_data += input_data * weight_data;
                            }
                        }
                    }
                    output[((b * oc + o_c) * oh + o_h) * ow + o_w] = result_data + bias[o_c];
                }
            }
        }
    }
}

// test depthwise
static void test_depthwise(int batch, int channel, int ih, int iw, int kh, int kw,
                           int stride, int pad_h, int pad_w, int dilation) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 1024*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    struct ggml_context * ctx = ggml_init(params);

    int oh, ow;
    oh = (ih + 2 * pad_h - (kh - 1) * dilation - 1) / stride + 1;
    ow = (iw + 2 * pad_w - (kw - 1) * dilation - 1) / stride + 1;

    int64_t ne1[4] = {kw, kh, channel, 1};
    int64_t ne2[4] = {iw, ih, channel, batch};

    struct ggml_tensor * a = get_random_tensor(ctx, 4, ne1, -1, +1);
    struct ggml_tensor * b = get_random_tensor(ctx, 4, ne2, -1, +1);
    ggml_set_param(ctx, a);
    // ggml_set_param(ctx, b);

    std::vector<float> output(ow*oh*channel*batch, 0);
    // int s0 = 1, s1 = 1, p0 = 1, p1 = 1, d0 = 1, d1 = 1;
    std::vector<float> bias(channel, 0);
    reference_conv2d((float*)(b->data), (float*)(a->data), bias.data(), output, batch, channel, channel, ih, iw,
                     pad_h, pad_w, kh, kw, stride, dilation, channel);
    struct ggml_tensor * e = ggml_conv_depthwise_2d(
        ctx, ggml_cast(ctx, a, GGML_TYPE_F16), b, stride, stride, pad_w, pad_h, dilation, dilation);


    struct ggml_cgraph * ge = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
    ggml_build_forward_expand(ge, e);
    ggml_graph_reset(ge);

    ggml_graph_compute_with_ctx(ctx, ge, /*n_threads*/ 1);

    // for(int i = 0; i < 10; i++){
    //     const float fe = ggml_get_f32_1d(e, i);
    //     printf("%s: e = %.4f\n", __func__, fe);
    //     printf("%s: e = %.4f\n", "gt", output[i]);
    // }
    bool res = true;
    res = output.size() == ggml_nelements(e);
    if (res == false) {
        printf("output.size() = %d, ggml_nelements(e) = %d\n", output.size(), ggml_nelements(e));
    }
    assert (res);

    for(int i = 0; i < output.size(); i++){
        const float fe = ggml_get_f32_1d(e, i);
        const float gt = output[i];
        res = abs(gt - fe) < 1e-8 + 1e-5 * abs(fe);
        assert(res);
    }
    ggml_free(ctx);
    printf("depthwise TEST PASSED with input: [%d, %d, %d, %d], kernel size: [%d, %d], stride: %d, dilation: %d, pad_h: %d, pad_w: %d\n",
            batch, channel, ih, iw, kh, kw, stride, dilation, pad_h, pad_w);
}

// test permute_cpy
static void reference_permute_0_2_3_1(const float* input, float* output,
                              const std::vector<int>& input_shape) {
    int dim0 = input_shape[0];
    int dim1 = input_shape[1];
    int dim2 = input_shape[2];
    int dim3 = input_shape[3];
    int dst_dim0 = dim0;
    int dst_dim2 = dim1;
    int dst_dim3 = dim2;
    int dst_dim1 = dim3;
    for (int i=0; i< dst_dim0; i++) {
        int stride0 = i*dst_dim1*dst_dim2*dst_dim3;
        for (int l=0; l< dst_dim1; l++) {
            int stride1 = l*dst_dim2*dst_dim3;
            for (int j=0; j< dst_dim2; j++) {
                int stride2 = j*dst_dim3;
                for (int k=0; k< dst_dim3; k++) {
                  output[stride0+stride1+stride2+k] = input[i*dim1*dim2*dim3+j*dim3*dim2+k*dim3+l];
                }
            }
        }
    }
}


static void reference_permute_0_3_1_2(const float* input, float* output,
                              const std::vector<int>& input_shape) {
    int dim0 = input_shape[0];
    int dim1 = input_shape[1];
    int dim2 = input_shape[2];
    int dim3 = input_shape[3];
    int dst_dim0 = dim0;
    int dst_dim3 = dim1;
    int dst_dim1 = dim2;
    int dst_dim2 = dim3;
    for (int i=0; i< dst_dim0; i++) {
        int stride0 = i*dst_dim1*dst_dim2*dst_dim3;
        for (int l=0; l< dst_dim1; l++) {
            int stride1 = l*dst_dim2*dst_dim3;
            for (int j=0; j< dst_dim2; j++) {
                int stride2 = j*dst_dim3;
                for (int k=0; k< dst_dim3; k++) {
                  output[stride0+stride1+stride2+k] = input[i*dim1*dim2*dim3+l*dim3+k*dim2*dim3+j];
                }
            }
        }
    }
}

static void reference_permute_0_1_3_2(const float* input, float* output,
                              const std::vector<int>& input_shape) {
    int dim0 = input_shape[0];
    int dim1 = input_shape[1];
    int dim2 = input_shape[2];
    int dim3 = input_shape[3];
    int dst_dim0 = dim0;
    int dst_dim1 = dim1;
    int dst_dim3 = dim2;
    int dst_dim2 = dim3;
    for (int i=0; i< dst_dim0; i++) {
        int stride0 = i*dst_dim1*dst_dim2*dst_dim3;
        for (int l=0; l< dst_dim1; l++) {
            int stride1 = l*dst_dim2*dst_dim3;
            for (int j=0; j< dst_dim2; j++) {
                int stride2 = j*dst_dim3;
                for (int k=0; k< dst_dim3; k++) {
                  output[stride0+stride1+stride2+k] = input[i*dim1*dim2*dim3+l*dim2*dim3+k*dim3+j];
                }
            }
        }
    }
}

static void test_permute_cpy(int batch, int channel, int ih, int iw,
                             int axis0, int axis1, int axis2, int axis3) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 1024*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    struct ggml_context * ctx = ggml_init(params);

    int64_t ne1[4] = {iw, ih, channel, batch};

    struct ggml_tensor * a = get_random_tensor(ctx, 4, ne1, -1, +1);
    // ggml_set_param(ctx, a);

    std::vector<float> output(iw*ih*channel*batch, 0);
    std::vector<int> input_shape = {batch, channel, ih, iw};
    if (axis0 == 0 && axis1 == 2 && axis2 == 3 && axis3 == 1)
        reference_permute_0_2_3_1((float*)(a->data), output.data(), input_shape);
    else if (axis0 == 0 && axis1 == 3 && axis2 == 1 && axis3 == 2)
        reference_permute_0_3_1_2((float*)(a->data), output.data(), input_shape);
    else if (axis0 == 0 && axis1 == 1 && axis2 == 3 && axis3 == 2)
        reference_permute_0_1_3_2((float*)(a->data), output.data(), input_shape);
    else
        assert(false);

    int reverse_axis3 = 3 - axis0;
    int reverse_axis2 = 3 - axis1;
    int reverse_axis1 = 3 - axis2;
    int reverse_axis0 = 3 - axis3;
    // // show axis and reverse_axis map info
    // printf("axis: %d, %d, %d, %d\n", axis0, axis1, axis2, axis3);
    // printf("reverse_axis: %d, %d, %d, %d\n", reverse_axis0, reverse_axis1, reverse_axis2, reverse_axis3);

    struct ggml_tensor * e = ggml_cont(ctx,ggml_permute(
        ctx, a, reverse_axis0, reverse_axis1, reverse_axis2, reverse_axis3));

    struct ggml_cgraph * ge = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
    ggml_build_forward_expand(ge, e);
    ggml_graph_reset(ge);

    ggml_graph_compute_with_ctx(ctx, ge, /*n_threads*/ 1);

    // // show src and dst
    // printf("src: ");
    // for (int i=0; i< iw*ih*channel*batch; i++) {
    //     printf("%.3f ", ((float*)(a->data))[i]);
    // }
    // printf("\n");
    // printf("dst: ");
    // for (int i=0; i< iw*ih*channel*batch; i++) {
    //     printf("%.3f ", output[i]);
    // }
    // printf("\n");
    // // ggml result
    // printf("ggml: ");
    // for(int i = 0; i < output.size(); i++){
    //     const float fe = ggml_get_f32_1d(e, i);
    //     printf("%.3f ", fe);
    // }


    bool res = true;
    for(int i = 0; i < output.size(); i++){
        const float fe = ggml_get_f32_1d(e, i);
        const float gt = output[i];
        res = abs(gt - fe) < 1e-8 + 1e-5 * abs(fe);
        assert(res);
    }
    ggml_free(ctx);
    printf("permute_cpy TEST PASSED with input: [%d, %d, %d, %d], axis: %d, %d, %d, %d\n",
            batch, channel, ih, iw, axis0, axis1, axis2, axis3);
}

// test gav_pool_2d
static void reference_gav_pool_2d(const float* input, float* output,
                                  int batch, int channel, int ih, int iw) {
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channel; ++c) {
            float sum = 0;
            for (int h = 0; h < ih; ++h) {
                for (int w = 0; w < iw; ++w) {
                    sum += input[((b * channel + c) * ih + h) * iw + w];
                }
            }
            output[b * channel + c] = sum / (ih * iw);
        }
    }
}

static void test_gav_pool_2d(int batch, int channel, int ih, int iw) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 1024*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    struct ggml_context * ctx = ggml_init(params);

    int64_t ne1[4] = {iw, ih, channel, batch};

    struct ggml_tensor * a = get_random_tensor(ctx, 4, ne1, -1, +1);
    // ggml_set_param(ctx, a);

    std::vector<float> output(channel*batch, 0);
    reference_gav_pool_2d((float*)(a->data), output.data(), batch, channel, ih, iw);

    struct ggml_tensor * e = ggml_pool_2d(
        ctx, a, GGML_OP_POOL_AVG, iw, ih, iw, ih, 0, 0);

    struct ggml_cgraph * ge = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
    ggml_build_forward_expand(ge, e);
    ggml_graph_reset(ge);

    ggml_graph_compute_with_ctx(ctx, ge, /*n_threads*/ 1);

    // printData((float*)(a->data), batch, channel, ih, iw, "input");
    // printData(output.data(), batch, channel, 1, 1, "reference");
    // printData((float*)(e->data), batch, channel, 1, 1, "ggml");

    bool res = true;
    for(int i = 0; i < output.size(); i++){
        const float fe = ggml_get_f32_1d(e, i);
        const float gt = output[i];
        res = abs(gt - fe) < 1e-8 + 1e-5 * abs(fe);
        assert(res);
    }
    ggml_free(ctx);
    printf("pool_2d TEST PASSED with input: [%d, %d, %d, %d]\n", batch, channel, ih, iw);
}

// test hardswish
static void reference_hardswish(const float* input, float* output, int batch, int channel, int ih, int iw) {
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channel; ++c) {
            for (int h = 0; h < ih; ++h) {
                for (int w = 0; w < iw; ++w) {
                    float x = input[((b * channel + c) * ih + h) * iw + w];
                    float y;
                    if (x <= -3) {
                        y = 0;
                    } else if (x >= 3) {
                        y = x;
                    } else {
                        y = x * (x + 3) / 6;
                    }
                    output[((b * channel + c) * ih + h) * iw + w] = y;
                }
            }
        }
    }
}

static void test_hardswish(int batch, int channel, int ih, int iw) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 1024*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    struct ggml_context * ctx = ggml_init(params);

    int64_t ne1[4] = {iw, ih, channel, batch};

    struct ggml_tensor * a = get_random_tensor(ctx, 4, ne1, -5, +5);

    std::vector<float> output(iw*ih*channel*batch, 0);
    reference_hardswish((float*)(a->data), output.data(), batch, channel, ih, iw);

    // printData((float*)(a->data), batch, channel, ih, iw, "input");
    // printData(output.data(), batch, channel, ih, iw, "reference");

    struct ggml_tensor * e = ggml_hardswish(ctx, a);

    struct ggml_cgraph * ge = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
    ggml_build_forward_expand(ge, e);
    ggml_graph_reset(ge);

    ggml_graph_compute_with_ctx(ctx, ge, /*n_threads*/ 1);

    // printData((float*)(e->data), batch, channel, ih, iw, "ggml");

    bool res = true;
    for(int i = 0; i < output.size(); i++){
        const float fe = ggml_get_f32_1d(e, i);
        const float gt = output[i];
        res = abs(gt - fe) < 1e-8 + 1e-5 * abs(fe);
        assert(res);
    }
    ggml_free(ctx);
    printf("hardswish TEST PASSED with input: [%d, %d, %d, %d]\n", batch, channel, ih, iw);
}

// test layer norm
static void reference_layer_norm(const float* input, float* output, int outer, int inner, float eps) {
    for (int i = 0; i < outer; ++i) {
        float sum = 0;
        for (int j = 0; j < inner; ++j) {
            sum += input[i * inner + j];
        }
        float mean = sum / inner;
        float square_sum = 0;
        for (int j = 0; j < inner; ++j) {
            float diff = input[i * inner + j] - mean;
            square_sum += diff * diff;
        }
        float variance = square_sum / inner;
        float inv_std = 1 / sqrt(variance + eps);
        for (int j = 0; j < inner; ++j) {
            output[i * inner + j] = (input[i * inner + j] - mean) * inv_std;
        }
    }
}

static void test_norm(int batch, int channel, int ih, int iw) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 1024*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };
    float eps = 1e-6;

    struct ggml_context * ctx = ggml_init(params);

    int64_t ne1[4] = {iw, ih, channel, batch};

    struct ggml_tensor * a = get_random_tensor(ctx, 4, ne1, -1, +1);

    std::vector<float> output(iw*ih*channel*batch, 0);
    int outer = batch * channel * ih;
    int inner = iw;
    reference_layer_norm((float*)(a->data), output.data(), outer, inner, eps);

    // printData((float*)(a->data), batch, channel, ih, iw, "input");
    // printData(output.data(), batch, channel, ih, iw, "reference");

    struct ggml_tensor * e = ggml_norm(
        ctx, a, eps);

    struct ggml_cgraph * ge = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
    ggml_build_forward_expand(ge, e);
    ggml_graph_reset(ge);

    ggml_graph_compute_with_ctx(ctx, ge, /*n_threads*/ 1);

    // printData((float*)(e->data), batch, channel, ih, iw, "ggml");

    bool res = true;
    for(int i = 0; i < output.size(); i++){
        const float fe = ggml_get_f32_1d(e, i);
        const float gt = output[i];
        // res = abs(gt - fe) < 1e-8 + 1e-5 * abs(fe);
        res = abs(gt - fe) < 1e-5;
        assert(res);
    }
    ggml_free(ctx);
    printf("layer norm TEST PASSED with input: [%d, %d, %d, %d]\n", batch, channel, ih, iw);
}

// test gelu
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
static void reference_gelu(const float* input, float* output, int batch, int channel, int ih, int iw) {
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channel; ++c) {
            for (int h = 0; h < ih; ++h) {
                for (int w = 0; w < iw; ++w) {
                    float x = input[((b * channel + c) * ih + h) * iw + w];
                    float y = 0.5 * x * (1 + tanh(sqrt(2 / 3.1415926) * (x + 0.044715 * x * x * x)));
                    output[((b * channel + c) * ih + h) * iw + w] = y;
                }
            }
        }
    }
}

static void test_gelu(int batch, int channel, int ih, int iw) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 1024*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    int64_t ne1[4] = {iw, ih, channel, batch};

    struct ggml_tensor * a = get_random_tensor(ctx, 4, ne1, -1, +1);

    std::vector<float> output(iw*ih*channel*batch, 0);
    reference_gelu((float*)(a->data), output.data(), batch, channel, ih, iw);

    // printData((float*)(a->data), batch, channel, ih, iw, "input");
    // printData(output.data(), batch, channel, ih, iw, "reference");

    struct ggml_tensor * e = ggml_gelu(ctx, a);

    struct ggml_cgraph * ge = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
    ggml_build_forward_expand(ge, e);
    ggml_graph_reset(ge);

    ggml_graph_compute_with_ctx(ctx, ge, /*n_threads*/ 1);

    // printData((float*)(e->data), batch, channel, ih, iw, "ggml");

    bool res = true;
    for(int i = 0; i < output.size(); i++){
        const float fe = ggml_get_f32_1d(e, i);
        const float gt = output[i];
        // res = abs(gt - fe) < 1e-8 + 1e-5 * abs(fe);
        res = abs(gt - fe) < 1e-3;
        if (res == false) {
            printf("i = %d, gt = %.4f, fe = %.4f\n", i, gt, fe);
        }
        assert(res);
    }
    ggml_free(ctx);
    printf("gelu TEST PASSED with input: [%d, %d, %d, %d]\n", batch, channel, ih, iw);
}

// test conv_2d
static void test_conv_2d(int batch, int channel, int ih, int iw, int kh, int kw,
                         int out_channel, int stride, int pad_h, int pad_w, int dilation) {
    struct ggml_init_params params = {
        /* .mem_size   = */ 1024*1024*1024,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ false,
    };

    struct ggml_context * ctx = ggml_init(params);

    int oh, ow;
    oh = (ih + 2 * pad_h - (kh - 1) * dilation - 1) / stride + 1;
    ow = (iw + 2 * pad_w - (kw - 1) * dilation - 1) / stride + 1;

    int64_t ne1[4] = {kw, kh, channel, out_channel};
    int64_t ne2[4] = {iw, ih, channel, batch};

    struct ggml_tensor * a = get_random_tensor(ctx, 4, ne1, -1, +1);
    struct ggml_tensor * b = get_random_tensor(ctx, 4, ne2, -1, +1);
    struct ggml_tensor * a_f16 = get_f16_tensor(ctx, a);

    std::vector<float> output(ow*oh*out_channel*batch, 0);
    // int s0 = 1, s1 = 1, p0 = 1, p1 = 1, d0 = 1, d1 = 1;
    std::vector<float> bias(channel, 0);
    reference_conv2d((float*)(b->data), (float*)(a->data), bias.data(), output,
                     batch, channel, out_channel, ih, iw,
                     pad_h, pad_w, kh, kw, stride, dilation, 1);

    // printData((float*)(a->data), out_channel, channel, kh, kw, "weight");
    // printData((float*)(b->data), batch, channel, ih, iw, "input");
    // printData(output.data(), batch, out_channel, oh, ow, "reference");

    struct ggml_tensor * e = ggml_conv_2d(
        ctx, a_f16, b, stride, stride, pad_w, pad_h, dilation, dilation);

    struct ggml_cgraph * ge = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
    ggml_build_forward_expand(ge, e);
    ggml_graph_reset(ge);

    ggml_graph_compute_with_ctx(ctx, ge, /*n_threads*/ 1);

    // printData((float*)(e->data), batch, out_channel, oh, ow, "ggml");

    bool res = true;

    assert (output.size() == ggml_nelements(e));

    for(int i = 0; i < output.size(); i++){
        const float fe = ggml_get_f32_1d(e, i);
        const float gt = output[i];
        // res = abs(gt - fe) < 1e-8 + 1e-5 * abs(fe);
        // f16 precision is not enough, to ease restriction
        res = abs(gt - fe) < 1e-3 || abs((gt - fe)/gt) < 0.05;
        if (res == false) {
            printf("i = %d, gt = %.4f, fe = %.4f, abs(gt-fe): %.4f, abs((gt - fe)/gt): %.4f\n",
                    i, gt, fe, abs(gt - fe), abs((gt - fe)/gt));
        }
        assert(res);
    }
    ggml_free(ctx);
    printf("conv TEST PASSED with input: [%d, %d, %d, %d], kernel size: [%d, %d], stride: %d, dilation: %d, pad_h: %d, pad_w: %d\n",
            batch, channel, ih, iw, kh, kw, stride, dilation, pad_h, pad_w);
}


static std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), buf.size());
}

static struct ggml_tensor * get_tensor(struct ggml_context * ctx, const std::string & name) {
    struct ggml_tensor * cur = ggml_get_tensor(ctx, name.c_str());
    if (!cur) {
        throw std::runtime_error(format("%s: unable to find tensor %s\n", __func__, name.c_str()));
    }

    return cur;
}


void read_data(float* result, int size, std::string filepath,
                                    char delim) {
  std::ifstream file_handle(filepath.c_str());
  std::string line;
  int curr_count = 0;
  if (file_handle.is_open()) {
    while (getline(file_handle, line)) {
      std::stringstream ss(line);
      while (getline(ss, line, delim)) {  // split line content
        result[curr_count++] = std::stof(line);
      }
    }
    file_handle.close();
    assert(size == curr_count);
  } else {
    printf("file:%s can't open normally!\n", filepath.c_str());
  }
}

bool dumpFile(const float* data_ptr, int elementSize,
              const std::string& filePath) {
  std::ofstream outputOs(filePath.c_str());
  if (outputOs.is_open()) {
    for (int i = 0; i < elementSize; i++) {
    outputOs << data_ptr[i] << "\n";
    }
    outputOs.close();
    return true;
  }
  return false;
}

struct ldp_model {
    // MobileVLM projection
    struct ggml_tensor * mm_model_mlp_1_w;
    struct ggml_tensor * mm_model_mlp_1_b;
    struct ggml_tensor * mm_model_mlp_3_w;
    struct ggml_tensor * mm_model_mlp_3_b;
    struct ggml_tensor * mm_model_block_1_block_0_0_w;
    struct ggml_tensor * mm_model_block_1_block_0_1_w;
    struct ggml_tensor * mm_model_block_1_block_0_1_b;
    struct ggml_tensor * mm_model_block_1_block_1_fc1_w;
    struct ggml_tensor * mm_model_block_1_block_1_fc1_b;
    struct ggml_tensor * mm_model_block_1_block_1_fc2_w;
    struct ggml_tensor * mm_model_block_1_block_1_fc2_b;
    struct ggml_tensor * mm_model_block_1_block_2_0_w;
    struct ggml_tensor * mm_model_block_1_block_2_1_w;
    struct ggml_tensor * mm_model_block_1_block_2_1_b;
    struct ggml_tensor * mm_model_block_2_block_0_0_w;
    struct ggml_tensor * mm_model_block_2_block_0_1_w;
    struct ggml_tensor * mm_model_block_2_block_0_1_b;
    struct ggml_tensor * mm_model_block_2_block_1_fc1_w;
    struct ggml_tensor * mm_model_block_2_block_1_fc1_b;
    struct ggml_tensor * mm_model_block_2_block_1_fc2_w;
    struct ggml_tensor * mm_model_block_2_block_1_fc2_b;
    struct ggml_tensor * mm_model_block_2_block_2_0_w;
    struct ggml_tensor * mm_model_block_2_block_2_1_w;
    struct ggml_tensor * mm_model_block_2_block_2_1_b;
    // peg projection
    struct ggml_tensor * mm_mlp_0_w;
    struct ggml_tensor * mm_mlp_0_b;
    struct ggml_tensor * mm_mlp_2_w;
    struct ggml_tensor * mm_mlp_2_b;
    struct ggml_tensor * mm_peg_proj_0_w;
    struct ggml_tensor * mm_peg_proj_0_b;
    struct ggml_tensor * mm_peg_ls_zeta;
};

struct ldp_ctx {
    struct ggml_context * ggml_ctx;
    struct gguf_context * gguf_ctx;
    struct ldp_model model;

    std::vector<uint8_t> buf_compute_meta;
    // memory buffers to evaluate the model
    ggml_backend_buffer_t params_buffer = NULL;
    ggml_backend_buffer_t compute_buffer = NULL;
    ggml_backend_t backend = NULL;
    ggml_gallocr_t compute_alloc = NULL;
};

static ggml_cgraph* build_graph(ldp_ctx* ctx, int batch, int n_patch, int n_embd,
                                float eps, const std::string& inp_path) {
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    std::vector<float> inp_data(n_embd * n_patch * n_patch * batch, 0);
    read_data(inp_data.data(), n_embd * n_patch * n_patch * batch, inp_path, ' ');

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * embeddings = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, n_patch*n_patch, batch);

    // ggml_allocr_alloc(ctx->compute_alloc, embeddings);

    // GGML_ASSERT(ggml_nbytes(embeddings) == inp_data.size() * sizeof(float));

    // if (!ggml_allocr_is_measure(ctx->compute_alloc)) {
    //     ggml_backend_tensor_set(embeddings, inp_data.data(), 0, ggml_nbytes(embeddings));
    // }
    ggml_set_name(embeddings, "embeddings");
    ggml_set_input(embeddings);
    
    ggml_backend_tensor_set(embeddings, inp_data.data(), 0, ggml_nbytes(embeddings));

    const auto & model = ctx->model;

    struct ggml_tensor * mlp_1 = ggml_mul_mat(ctx0, model.mm_model_mlp_1_w, embeddings);
    mlp_1 = ggml_add(ctx0, mlp_1, model.mm_model_mlp_1_b);
    mlp_1 = ggml_gelu(ctx0, mlp_1);
    struct ggml_tensor * mlp_3 = ggml_mul_mat(ctx0, model.mm_model_mlp_3_w, mlp_1);
    mlp_3 = ggml_add(ctx0, mlp_3, model.mm_model_mlp_3_b);
    // transpose from [1, 576, 2048] --> [1, 24, 24, 2048] --> [1, 2048, 24, 24]
    // mlp_3.ne = [2048, 576, 1, 1]
    mlp_3 = ggml_reshape_4d(ctx0, mlp_3, mlp_3->ne[0], n_patch, n_patch, mlp_3->ne[3]);
    // mlp_3 shape = [1, 24, 24, 2048],  ne = [2048, 24, 24, 1]
    // permute logic is src idxs 0,1,2,3 perm to dst idxs
    mlp_3 = ggml_cont(ctx0, ggml_permute(ctx0, mlp_3, 2, 0, 1, 3));
    // mlp_3 shape = [1, 2048, 24, 24],  ne = [24, 24, 2048, 1]

    // block 1
    struct ggml_tensor * block_1 = nullptr;
    {
        // stride = 1, padding = 1, bias is nullptr
        block_1 = ggml_conv_depthwise_2d(ctx0, model.mm_model_block_1_block_0_0_w, mlp_3, 1, 1, 1, 1, 1, 1);

        // layer norm
        // // block_1 shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1]
        block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
        // block_1 shape = [1, 24, 24, 2048], ne = [2048, 24, 24, 1]
        block_1 = ggml_norm(ctx0, block_1, eps);
        block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_1_block_0_1_w), model.mm_model_block_1_block_0_1_b);
        block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));

        // block_1 shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1]
        // hardswish
        struct ggml_tensor * block_1_hw = ggml_hardswish(ctx0, block_1);

        block_1 = ggml_pool_2d(ctx0, block_1_hw, GGML_OP_POOL_AVG, block_1_hw->ne[0], block_1_hw->ne[1], block_1_hw->ne[0], block_1_hw->ne[1], 0, 0);
        // block_1 shape = [1, 2048, 1, 1], ne = [1, 1, 2048, 1]
        // pointwise conv
        block_1 = ggml_reshape_2d(ctx0, block_1, block_1->ne[0]*block_1->ne[1]*block_1->ne[2], block_1->ne[3]);
        block_1 = ggml_mul_mat(ctx0, model.mm_model_block_1_block_1_fc1_w, block_1);
        block_1 = ggml_add(ctx0, block_1, model.mm_model_block_1_block_1_fc1_b);
        block_1 = ggml_relu(ctx0, block_1);
        block_1 = ggml_mul_mat(ctx0, model.mm_model_block_1_block_1_fc2_w, block_1);
        block_1 = ggml_add(ctx0, block_1, model.mm_model_block_1_block_1_fc2_b);
        block_1 = ggml_hardsigmoid(ctx0, block_1);
        // block_1_hw shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1], block_1 shape = [1, 2048], ne = [2048, 1, 1, 1]
        block_1 = ggml_reshape_4d(ctx0, block_1, 1, 1, block_1->ne[0], block_1->ne[1]);
        block_1 = ggml_mul(ctx0, block_1_hw, block_1);

        // block_1 shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1]
        struct ggml_tensor* block_2_0_w_4d = ggml_reshape_4d(ctx0, model.mm_model_block_1_block_2_0_w, 1, 1,
                model.mm_model_block_1_block_2_0_w->ne[0], model.mm_model_block_1_block_2_0_w->ne[1]);
        block_1 = ggml_conv_2d(ctx0, block_2_0_w_4d, block_1, 1, 1, 0, 0, 1, 1);

        // layernorm
        block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
        // block_1 shape = [1, 24, 24, 2048], ne = [2048, 24, 24, 1]
        block_1 = ggml_norm(ctx0, block_1, eps);
        block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_1_block_2_1_w), model.mm_model_block_1_block_2_1_b);
        block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
        // block1 shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1]
        // residual
        block_1 = ggml_add(ctx0, mlp_3, block_1);
    }

    // block_2
    {
        // to implement depthwise op
        // stride = 2
        block_1 = ggml_conv_depthwise_2d(ctx0, model.mm_model_block_2_block_0_0_w, block_1, 2, 2, 1, 1, 1, 1);

        // block_1 shape = [1, 2048, 12, 12], ne = [12, 12, 2048, 1]
        // layer norm
        block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
        // block_1 shape = [1, 12, 12, 2048], ne = [2048, 12, 12, 1]
        block_1 = ggml_norm(ctx0, block_1, eps);
        block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_2_block_0_1_w), model.mm_model_block_2_block_0_1_b);
        block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
        // block_1 shape = [1, 2048, 12, 12], ne = [12, 12, 2048, 1]
        // hardswish
        struct ggml_tensor * block_1_hw = ggml_hardswish(ctx0, block_1);

        // not sure the parameters is right for globalAvgPooling
        block_1 = ggml_pool_2d(ctx0, block_1_hw, GGML_OP_POOL_AVG, block_1_hw->ne[0], block_1_hw->ne[1], block_1_hw->ne[0], block_1_hw->ne[1], 0, 0);
        // block_1 shape = [1, 2048, 1, 1], ne = [1, 1, 2048, 1]
        // pointwise conv
        block_1 = ggml_reshape_2d(ctx0, block_1, block_1->ne[0]*block_1->ne[1]*block_1->ne[2], block_1->ne[3]);
        block_1 = ggml_mul_mat(ctx0, model.mm_model_block_2_block_1_fc1_w, block_1);
        block_1 = ggml_add(ctx0, block_1, model.mm_model_block_2_block_1_fc1_b);
        block_1 = ggml_relu(ctx0, block_1);
        block_1 = ggml_mul_mat(ctx0, model.mm_model_block_2_block_1_fc2_w, block_1);
        block_1 = ggml_add(ctx0, block_1, model.mm_model_block_2_block_1_fc2_b);
        block_1 = ggml_hardsigmoid(ctx0, block_1);

        // block_1_hw shape = [1, 2048, 12, 12], ne = [12, 12, 2048, 1], block_1 shape = [1, 2048, 1, 1], ne = [1, 1, 2048, 1]
        block_1 = ggml_reshape_4d(ctx0, block_1, 1, 1, block_1->ne[0], block_1->ne[1]);
        block_1 = ggml_mul(ctx0, block_1_hw, block_1);
        // block_1 shape = [1, 2048, 12, 12], ne = [12, 12, 2048, 1]
        struct ggml_tensor* block_2_0_w_4d = ggml_reshape_4d(ctx0, model.mm_model_block_2_block_2_0_w, 1, 1,
                model.mm_model_block_2_block_2_0_w->ne[0], model.mm_model_block_1_block_2_0_w->ne[1]);
        block_1 = ggml_conv_2d(ctx0, block_2_0_w_4d, block_1, 1, 1, 0, 0, 1, 1);
        // layernorm
        block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
        // block_1 shape = [1, 12, 12, 2048], ne = [2048, 12, 12, 1]
        block_1 = ggml_norm(ctx0, block_1, eps);
        block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_2_block_2_1_w), model.mm_model_block_2_block_2_1_b);
        block_1 = ggml_reshape_3d(ctx0, block_1, block_1->ne[0], block_1->ne[1] * block_1->ne[2], block_1->ne[3]);
        // block_1 shape = [1, 144, 2048], ne = [2048, 144, 1]
    }
    // build the graph
    ggml_build_forward_expand(gf, block_1);

    ggml_free(ctx0);

    return gf;
}

static ggml_tensor* inference(ldp_ctx* ctx, int batch, int n_patch, int n_embd,
                                float eps, const std::string& inp_path, int n_threads) {
    // // reset alloc buffer to clean the memory from previous invocations
    // ggml_allocr_reset(ctx->compute_alloc);

    // build the inference graph
    ggml_cgraph * gf = build_graph(ctx, batch, n_patch, n_embd, eps, inp_path);
    // ggml_allocr_alloc_graph(ctx->compute_alloc, gf);
    ggml_gallocr_alloc_graph(ctx->compute_alloc, gf);
    if (ggml_backend_is_cpu(ctx->backend)) {
        ggml_backend_cpu_set_n_threads(ctx->backend, n_threads);
    }
    ggml_backend_graph_compute(ctx->backend, gf);
    // the last node is the embedding tensor
    struct ggml_tensor * embeddings = gf->nodes[gf->n_nodes - 1];
    return embeddings;
}

// test ldp
static void test_ldp() {

    struct ldp_ctx* ctx = new ldp_ctx;
    int n_threads = 1;
    int batch = 1;
    int n_patch = 24;
    int n_embd = 1024;
    float eps = 1e-5;
    int ldp_tensor_num = 24;
    std::string inp_path = "/Users/cxt/model/llm/mobileVLM/ldp_model/ldp_io/ldp_input.txt";
    std::string out_dir = "/Users/cxt/model/llm/mobileVLM/ldp_model/ldp_layers/llama.cpp/";
    // load parmas, construct graph
    std::string fname = "/Users/cxt/model/llm/mobileVLM/MobileVLM-1.7B_processed/mmproj-model-f16.gguf";

    struct ggml_context * meta = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &meta,
    };

    struct gguf_context * gguf_ctx = gguf_init_from_file(fname.c_str(), params);
    if (!gguf_ctx) {
        throw std::runtime_error(format("%s: failed to load CLIP model from %s. Does this file exist?\n", __func__, fname.c_str()));
    }

    std::string KEY_DESCRIPTION = "general.description";
    std::string KEY_NAME = "general.name";
    std::string KEY_PROJ_TYPE = "clip.projector_type";
    const int n_tensors = gguf_get_n_tensors(gguf_ctx);
    const int n_kv = gguf_get_n_kv(gguf_ctx);
    const int idx_desc = gguf_find_key(gguf_ctx, KEY_DESCRIPTION.c_str());
    const std::string description = gguf_get_val_str(gguf_ctx, idx_desc);
    const int idx_name = gguf_find_key(gguf_ctx, KEY_NAME.c_str());
    if (idx_name != -1) { // make name optional temporarily as some of the uploaded models missing it due to a bug
        const std::string name = gguf_get_val_str(gguf_ctx, idx_name);
        printf("%s: model name:   %s\n", __func__, name.c_str());
    }
    int idx_proj_type = gguf_find_key(gguf_ctx, KEY_PROJ_TYPE.c_str());
    if (idx_proj_type != -1) {
        const std::string proj_type = gguf_get_val_str(gguf_ctx, idx_proj_type);
        printf("%s: projector name:   %s\n", __func__, proj_type.c_str());
    }
    printf("%s: description:  %s\n", __func__, description.c_str());
    printf("%s: GGUF version: %d\n", __func__, gguf_get_version(gguf_ctx));
    printf("%s: alignment:    %zu\n", __func__, gguf_get_alignment(gguf_ctx));
    printf("%s: n_tensors:    %d\n", __func__, n_tensors);
    printf("%s: n_kv:         %d\n", __func__, n_kv);
    printf("\n");

    // data
    int match_tensor_num = 0;
    size_t buffer_size = 0;
    {
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf_ctx, i);
            if (strncmp(name, "mm.model", 8) == 0) {
                match_tensor_num++;
                const size_t offset = gguf_get_tensor_offset(gguf_ctx, i);
                enum ggml_type type = gguf_get_tensor_type(gguf_ctx, i);
                struct ggml_tensor * cur = ggml_get_tensor(meta, name);
                size_t tensor_size = ggml_nbytes(cur);
                buffer_size += tensor_size;
                // printf("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape:[%d, %d, %d, %d], type: %d\n", __func__, i,
                //        ggml_n_dims(cur), cur->name, tensor_size, offset, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3], type);
            }
            else {
                continue;
            }
        }
    }

    printf("%s: match_tensor_num = %d, buffer_size: %.3f MB, %d Bytes\n",
        __func__, match_tensor_num, buffer_size/1024.0/1024.0, buffer_size);

    buffer_size += ldp_tensor_num * 128 /* CLIP PADDING */;


    std::vector<uint8_t>& buf_compute_meta = ctx->buf_compute_meta;
    // memory buffers to evaluate the model
    ggml_backend_buffer_t& params_buffer = ctx->params_buffer;
    ggml_backend_buffer_t& compute_buffer = ctx->compute_buffer;
    ggml_backend_t& backend = ctx->backend;
    // ggml_allocr *& compute_alloc = ctx->compute_alloc;


    struct ggml_context *& ggml_ctx = ctx->ggml_ctx;
    backend = ggml_backend_cpu_init();
    // params_buffer = ggml_backend_alloc_buffer(backend, buffer_size);
    params_buffer = ggml_backend_alloc_ctx_tensors(ggml_ctx, backend);

    // load tensors to ggml_ctx
    {
        std::vector<uint8_t> read_buf;
        struct ggml_init_params params = {
            /*.mem_size =*/ (ldp_tensor_num + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc =*/ true,
        };

        ggml_ctx = ggml_init(params);
        if (!ggml_ctx) {
            throw std::runtime_error(format("%s: failed to initialize ggml context\n", __func__));
        }
        std::ifstream fin(fname.c_str(), std::ios::binary);
        if (!fin) {
            throw std::runtime_error(format("%s: failed to open file\n", __func__));
        }
        // add tensors to context
        std::vector<int> ldp_tensor_idxs;
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf_ctx, i);
            // if name start with "mm.model", add it to ggml context
            if (strncmp(name, "mm.model", 8) == 0) {
                ldp_tensor_idxs.push_back(i);
                struct ggml_tensor * t = ggml_get_tensor(meta, name);
                struct ggml_tensor * cur = ggml_dup_tensor(ggml_ctx, t);
                ggml_set_name(cur, name);
                // printf("%s: ggml dup tensor name: %s\n", __func__, name);
            } else {
                continue;
            }
        }

        GGML_ASSERT(ldp_tensor_idxs.size() == 24);
        printf("%s: ldp_tensor_idxs size: %d\n", __func__, ldp_tensor_idxs.size());

        // alloc memory and offload data
        // ggml_allocr* alloc = ggml_allocr_new_from_buffer(params_buffer);
        for (int i = 0; i < ldp_tensor_idxs.size(); ++i) {
            int idx = ldp_tensor_idxs[i];
            const char * name = gguf_get_tensor_name(gguf_ctx, idx);
            struct ggml_tensor * cur = ggml_get_tensor(ggml_ctx, name);
            // ggml_allocr_alloc(alloc, cur);
            const size_t offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, idx);
            fin.seekg(offset, std::ios::beg);
            if (!fin) {
                printf("%s: failed to seek for tensor %s\n", __func__, name);
                return;
            }
            int num_bytes = ggml_nbytes(cur);
            if (ggml_backend_buffer_is_host(params_buffer)) {
                // for the CPU and Metal backend, we can read directly into the tensor
                fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(num_bytes);
                fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
                ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
            }
        }
        // ggml_allocr_free(alloc);
        fin.close();
    }

    // get tensor from ggml_ctx
    {
        struct ldp_model& model = ctx->model;
        model.mm_model_mlp_1_w = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_MLP, 1, "weight"));
        model.mm_model_mlp_1_b = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_MLP, 1, "bias"));
        model.mm_model_mlp_3_w = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_MLP, 3, "weight"));
        model.mm_model_mlp_3_b = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_MLP, 3, "bias"));
        model.mm_model_block_1_block_0_0_w = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_BLOCK, 1, 0, "0.weight"));
        model.mm_model_block_1_block_0_1_w = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_BLOCK, 1, 0, "1.weight"));
        model.mm_model_block_1_block_0_1_b = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_BLOCK, 1, 0, "1.bias"));
        model.mm_model_block_1_block_1_fc1_w = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc1.weight"));
        model.mm_model_block_1_block_1_fc1_b = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc1.bias"));
        model.mm_model_block_1_block_1_fc2_w = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc2.weight"));
        model.mm_model_block_1_block_1_fc2_b = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc2.bias"));
        model.mm_model_block_1_block_2_0_w = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_BLOCK, 1, 2, "0.weight"));
        model.mm_model_block_1_block_2_1_w = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_BLOCK, 1, 2, "1.weight"));
        model.mm_model_block_1_block_2_1_b = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_BLOCK, 1, 2, "1.bias"));
        model.mm_model_block_2_block_0_0_w = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_BLOCK, 2, 0, "0.weight"));
        model.mm_model_block_2_block_0_1_w = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_BLOCK, 2, 0, "1.weight"));
        model.mm_model_block_2_block_0_1_b = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_BLOCK, 2, 0, "1.bias"));
        model.mm_model_block_2_block_1_fc1_w = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc1.weight"));
        model.mm_model_block_2_block_1_fc1_b = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc1.bias"));
        model.mm_model_block_2_block_1_fc2_w = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc2.weight"));
        model.mm_model_block_2_block_1_fc2_b = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc2.bias"));
        model.mm_model_block_2_block_2_0_w = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_BLOCK, 2, 2, "0.weight"));
        model.mm_model_block_2_block_2_1_w = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_BLOCK, 2, 2, "1.weight"));
        model.mm_model_block_2_block_2_1_b = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_BLOCK, 2, 2, "1.bias"));
    }

    ggml_free(meta);

    // measure mem requirement and allocate
    {
        ctx->buf_compute_meta.resize(GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead());
        // ctx->compute_alloc = ggml_allocr_new_measure_from_backend(backend);
        ctx->compute_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_cgraph * gf = build_graph(ctx, batch, n_patch, n_embd, eps, inp_path);
        // size_t compute_memory_buffer_size = ggml_allocr_alloc_graph(ctx->compute_alloc, gf);
        // ggml_allocr_free(ctx->compute_alloc);
        // ctx->compute_buffer = ggml_backend_alloc_buffer(ctx->backend, compute_memory_buffer_size);
        // ctx->compute_alloc = ggml_allocr_new_from_buffer(ctx->compute_buffer);
        ggml_gallocr_reserve(ctx->compute_alloc, gf);
        size_t compute_memory_buffer_size = ggml_gallocr_get_buffer_size(ctx->compute_alloc, 0);
        printf("%s: compute allocated memory: %.2f MB\n", __func__, compute_memory_buffer_size /1024.0/1024.0);
    }

    // do inference
    {
        struct ggml_tensor * embeddings = inference(ctx, batch, n_patch, n_embd, eps, inp_path, n_threads);
        std::vector<float> result(ggml_nelements(embeddings));
        // copy the embeddings to the location passed by the user
        ggml_backend_tensor_get(embeddings, result.data(), 0, ggml_nbytes(embeddings));
        // save output
        const char * name = ggml_get_name(embeddings);
        std::string out_path = out_dir + std::string(name) + "_llama.cpp.txt";
        dumpFile(result.data(), result.size(), out_path);
        printf("%s: output tensor name: %s, shape: [%d, %d, %d, %d] dump to: %s\n",
                __func__, name, embeddings->ne[0], embeddings->ne[1],
                embeddings->ne[2], embeddings->ne[3], out_path.c_str());
    }

}

// 0202 projector v21
static ggml_cgraph* build_graph_peg(ldp_ctx* ctx, int batch, int n_patch, int n_embd,
                                float eps, const std::string& inp_path) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    std::vector<float> inp_data(n_embd * n_patch * n_patch * batch, 0);
    read_data(inp_data.data(), n_embd * n_patch * n_patch * batch, inp_path, ' ');

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * embeddings = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, n_patch*n_patch, batch);

    // ggml_allocr_alloc(ctx->compute_alloc, embeddings);

    // GGML_ASSERT(ggml_nbytes(embeddings) == inp_data.size() * sizeof(float));

    ggml_set_name(embeddings, "embeddings");
    ggml_set_input(embeddings);
    
    ggml_backend_tensor_set(embeddings, inp_data.data(), 0, ggml_nbytes(embeddings));

    const auto & model = ctx->model;

    struct ggml_tensor * mlp_0 = ggml_mul_mat(ctx0, model.mm_mlp_0_w, embeddings);
    mlp_0 = ggml_add(ctx0, mlp_0, model.mm_mlp_0_b);
    mlp_0 = ggml_gelu(ctx0, mlp_0);
    struct ggml_tensor * mlp_2 = ggml_mul_mat(ctx0, model.mm_mlp_2_w, mlp_0);
    mlp_2 = ggml_add(ctx0, mlp_2, model.mm_mlp_2_b);
    // transpose from [1, 576, 2048] --> [1, 2048, 576] --> [1, 2048, 24, 24]
    mlp_2 = ggml_cont(ctx0, ggml_permute(ctx0, mlp_2, 1, 0, 2, 3));
    mlp_2 = ggml_reshape_4d(ctx0, mlp_2, n_patch, n_patch, mlp_2->ne[1], mlp_2->ne[2]);

    // print_tensor_info(mlp_2, "mlp_2");

    // shape = [1, 2048, 12, 12]
    embeddings = ggml_pool_2d(ctx0, mlp_2, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);

    // shape = [1, 2048, 144]
    struct ggml_tensor* residual = ggml_reshape_3d(ctx0, embeddings, embeddings->ne[0]*embeddings->ne[1], embeddings->ne[2], embeddings->ne[3]);
    // shape = [1, 144, 2048]
    residual = ggml_cont(ctx0, ggml_permute(ctx0, residual, 1, 0, 2, 3));

    // depthwise conv
    embeddings = ggml_conv_depthwise_2d(ctx0, model.mm_peg_proj_0_w, embeddings, 1, 1, 1, 1, 1, 1);
    struct ggml_tensor* peg_bias = ggml_reshape_4d(ctx0, model.mm_peg_proj_0_b, 1, 1, model.mm_peg_proj_0_b->ne[0], 1);
    embeddings = ggml_add(ctx0, embeddings, peg_bias);
    // ls
    // shape = [1, 2048, 12, 12] --> [1, 2048, 144]
    embeddings = ggml_reshape_3d(ctx0, embeddings, embeddings->ne[0]*embeddings->ne[1], embeddings->ne[2], embeddings->ne[3]);
    // shape = [1, 144, 2048]
    embeddings = ggml_cont(ctx0, ggml_permute(ctx0, embeddings, 1, 0, 2, 3));
    embeddings = ggml_mul(ctx0, embeddings, model.mm_peg_ls_zeta);
    // add
    embeddings = ggml_add(ctx0, embeddings, residual);

    // build the graph
    ggml_build_forward_expand(gf, embeddings);

    ggml_free(ctx0);

    return gf;
}

static ggml_tensor* inference_peg(ldp_ctx* ctx, int batch, int n_patch, int n_embd,
                                float eps, const std::string& inp_path, int n_threads) {
    // // reset alloc buffer to clean the memory from previous invocations
    // ggml_allocr_reset(ctx->compute_alloc);

    // build the inference graph
    ggml_cgraph * gf = build_graph_peg(ctx, batch, n_patch, n_embd, eps, inp_path);
    // ggml_allocr_alloc_graph(ctx->compute_alloc, gf);
    ggml_gallocr_alloc_graph(ctx->compute_alloc, gf);
    if (ggml_backend_is_cpu(ctx->backend)) {
        ggml_backend_cpu_set_n_threads(ctx->backend, n_threads);
    }
    ggml_backend_graph_compute(ctx->backend, gf);
    // the last node is the embedding tensor
    struct ggml_tensor * embeddings = gf->nodes[gf->n_nodes - 1];
    return embeddings;
}

static void test_peg() {

    struct ldp_ctx* ctx = new ldp_ctx;
    int n_threads = 1;
    int batch = 1;
    int n_patch = 24;
    int n_embd = 1024;
    float eps = 1e-5;
    int ldp_tensor_num = 7;
    std::string inp_path = "/Users/cxt/model/llm/mobileVLM/0202/ldp_io/input.txt";
    std::string out_dir = "/Users/cxt/model/llm/mobileVLM/0202/ldp_layers/llama.cpp/";
    // load parmas, construct graph
    std::string fname = "/Users/cxt/model/llm/mobileVLM/0202/llava1.5-2.finetune/mmproj-model-f16.gguf";

    struct ggml_context * meta = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &meta,
    };

    struct gguf_context * gguf_ctx = gguf_init_from_file(fname.c_str(), params);
    if (!gguf_ctx) {
        throw std::runtime_error(format("%s: failed to load CLIP model from %s. Does this file exist?\n", __func__, fname.c_str()));
    }

    std::string KEY_DESCRIPTION = "general.description";
    std::string KEY_NAME = "general.name";
    std::string KEY_PROJ_TYPE = "clip.projector_type";
    const int n_tensors = gguf_get_n_tensors(gguf_ctx);
    const int n_kv = gguf_get_n_kv(gguf_ctx);
    const int idx_desc = gguf_find_key(gguf_ctx, KEY_DESCRIPTION.c_str());
    const std::string description = gguf_get_val_str(gguf_ctx, idx_desc);
    const int idx_name = gguf_find_key(gguf_ctx, KEY_NAME.c_str());
    if (idx_name != -1) { // make name optional temporarily as some of the uploaded models missing it due to a bug
        const std::string name = gguf_get_val_str(gguf_ctx, idx_name);
        printf("%s: model name:   %s\n", __func__, name.c_str());
    }
    int idx_proj_type = gguf_find_key(gguf_ctx, KEY_PROJ_TYPE.c_str());
    if (idx_proj_type != -1) {
        const std::string proj_type = gguf_get_val_str(gguf_ctx, idx_proj_type);
        printf("%s: projector name:   %s\n", __func__, proj_type.c_str());
    }
    printf("%s: description:  %s\n", __func__, description.c_str());
    printf("%s: GGUF version: %d\n", __func__, gguf_get_version(gguf_ctx));
    printf("%s: alignment:    %zu\n", __func__, gguf_get_alignment(gguf_ctx));
    printf("%s: n_tensors:    %d\n", __func__, n_tensors);
    printf("%s: n_kv:         %d\n", __func__, n_kv);
    printf("\n");

    // data
    int match_tensor_num = 0;
    size_t buffer_size = 0;
    {
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf_ctx, i);
            if (strncmp(name, "mm.peg", 6) == 0 || strncmp(name, "mm.mlp", 6) == 0) {
                match_tensor_num++;
                const size_t offset = gguf_get_tensor_offset(gguf_ctx, i);
                enum ggml_type type = gguf_get_tensor_type(gguf_ctx, i);
                struct ggml_tensor * cur = ggml_get_tensor(meta, name);
                size_t tensor_size = ggml_nbytes(cur);
                buffer_size += tensor_size;
                // printf("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape:[%d, %d, %d, %d], type: %d\n", __func__, i,
                //        ggml_n_dims(cur), cur->name, tensor_size, offset, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3], type);
            }
            else {
                continue;
            }
        }
    }

    printf("%s: match_tensor_num = %d, buffer_size: %.3f MB, %d Bytes\n",
        __func__, match_tensor_num, buffer_size/1024.0/1024.0, buffer_size);

    buffer_size += ldp_tensor_num * 128 /* CLIP PADDING */;


    std::vector<uint8_t>& buf_compute_meta = ctx->buf_compute_meta;
    // memory buffers to evaluate the model
    ggml_backend_buffer_t& params_buffer = ctx->params_buffer;
    ggml_backend_buffer_t& compute_buffer = ctx->compute_buffer;
    ggml_backend_t& backend = ctx->backend;


    struct ggml_context *& ggml_ctx = ctx->ggml_ctx;
    backend = ggml_backend_cpu_init();
    // params_buffer = ggml_backend_alloc_buffer(backend, buffer_size);
    params_buffer = ggml_backend_alloc_ctx_tensors(ggml_ctx, backend);

    // load tensors to ggml_ctx
    {
        std::vector<uint8_t> read_buf;
        struct ggml_init_params params = {
            /*.mem_size =*/ (ldp_tensor_num + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc =*/ true,
        };

        ggml_ctx = ggml_init(params);
        if (!ggml_ctx) {
            throw std::runtime_error(format("%s: failed to initialize ggml context\n", __func__));
        }
        std::ifstream fin(fname.c_str(), std::ios::binary);
        if (!fin) {
            throw std::runtime_error(format("%s: failed to open file\n", __func__));
        }
        // add tensors to context
        std::vector<int> ldp_tensor_idxs;
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gguf_ctx, i);
            // if name start with "mm.model", add it to ggml context
            if (strncmp(name, "mm.peg", 6) == 0 || strncmp(name, "mm.mlp", 6) == 0) {
                ldp_tensor_idxs.push_back(i);
                struct ggml_tensor * t = ggml_get_tensor(meta, name);
                struct ggml_tensor * cur = ggml_dup_tensor(ggml_ctx, t);
                ggml_set_name(cur, name);
                // printf("%s: ggml dup tensor name: %s\n", __func__, name);
            } else {
                continue;
            }
        }

        GGML_ASSERT(ldp_tensor_idxs.size() == 7);
        printf("%s: ldp_tensor_idxs size: %d\n", __func__, ldp_tensor_idxs.size());

        // alloc memory and offload data
        // ggml_allocr* alloc = ggml_allocr_new_from_buffer(params_buffer);
        for (int i = 0; i < ldp_tensor_idxs.size(); ++i) {
            int idx = ldp_tensor_idxs[i];
            const char * name = gguf_get_tensor_name(gguf_ctx, idx);
            struct ggml_tensor * cur = ggml_get_tensor(ggml_ctx, name);
            // ggml_allocr_alloc(alloc, cur);
            const size_t offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, idx);
            fin.seekg(offset, std::ios::beg);
            if (!fin) {
                printf("%s: failed to seek for tensor %s\n", __func__, name);
                return;
            }
            int num_bytes = ggml_nbytes(cur);
            if (ggml_backend_buffer_is_host(params_buffer)) {
                // for the CPU and Metal backend, we can read directly into the tensor
                fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(num_bytes);
                fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
                ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
            }
        }
        // ggml_allocr_free(alloc);
        fin.close();
    }

    // get tensor from ggml_ctx
    {
        struct ldp_model& model = ctx->model;
        model.mm_mlp_0_w = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_PEG_MLP, 0, "weight"));
        model.mm_mlp_0_b = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_PEG_MLP, 0, "bias"));
        model.mm_mlp_2_w = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_PEG_MLP, 2, "weight"));
        model.mm_mlp_2_b = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_PEG_MLP, 2, "bias"));
        model.mm_peg_proj_0_w = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_PEG, 0, "weight"));
        model.mm_peg_proj_0_b = get_tensor(ggml_ctx, format(TN_MVLM_PROJ_PEG, 0, "bias"));
        model.mm_peg_ls_zeta = get_tensor(ggml_ctx, "mm.peg.ls.zeta");
    }

    ggml_free(meta);

    // measure mem requirement and allocate
    {
        ctx->buf_compute_meta.resize(GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead());
        // ctx->compute_alloc = ggml_allocr_new_measure_from_backend(backend);
        ctx->compute_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        ggml_cgraph * gf = build_graph_peg(ctx, batch, n_patch, n_embd, eps, inp_path);
        // size_t compute_memory_buffer_size = ggml_allocr_alloc_graph(ctx->compute_alloc, gf);
        // ggml_allocr_free(ctx->compute_alloc);
        // ctx->compute_buffer = ggml_backend_alloc_buffer(ctx->backend, compute_memory_buffer_size);
        // ctx->compute_alloc = ggml_allocr_new_from_buffer(ctx->compute_buffer);
        ggml_gallocr_reserve(ctx->compute_alloc, gf);
        size_t compute_memory_buffer_size = ggml_gallocr_get_buffer_size(ctx->compute_alloc, 0);
        printf("%s: compute allocated memory: %.2f MB\n", __func__, compute_memory_buffer_size /1024.0/1024.0);
    }

    // do inference
    {
        struct ggml_tensor * embeddings = inference_peg(ctx, batch, n_patch, n_embd, eps, inp_path, n_threads);
        std::vector<float> result(ggml_nelements(embeddings));
        // copy the embeddings to the location passed by the user
        ggml_backend_tensor_get(embeddings, result.data(), 0, ggml_nbytes(embeddings));
        // save output
        const char * name = ggml_get_name(embeddings);
        std::string out_path = out_dir + std::string(name) + "_llama.cpp.txt";
        dumpFile(result.data(), result.size(), out_path);
        printf("%s: output tensor name: %s, shape: [%d, %d, %d, %d] dump to: %s\n",
                __func__, name, embeddings->ne[0], embeddings->ne[1],
                embeddings->ne[2], embeddings->ne[3], out_path.c_str());
    }

}



int main(void) {
    // test_depthwise(1, 2048, 24, 24, 3, 3, 1, 1, 1, 1);
    // test_depthwise(1, 2048, 24, 24, 3, 3, 2, 1, 1, 1);
    // test_permute_cpy(1, 24, 24, 2048, 0, 3, 1, 2);
    // test_permute_cpy(1, 2048, 24, 24, 0, 2, 3, 1);
    // test_permute_cpy(1, 12, 12, 2048, 0, 3, 1, 2);
    // test_permute_cpy(1, 2048, 12, 12, 0, 2, 3, 1);
    // test_permute_cpy(1, 1, 144, 2048, 0, 1, 3, 2);
    // test_gav_pool_2d(1, 3, 2, 2);
    // test_gav_pool_2d(1, 2048, 24, 24);
    // test_gav_pool_2d(1, 2048, 12, 12);
    // test_hardswish(1, 3, 2, 2);
    // test_hardswish(1, 2048, 24, 24);
    // test_hardswish(1, 2048, 12, 12);
    // test_norm(1, 2, 2, 3);
    // test_norm(1, 24, 24, 2048);
    // test_norm(1, 12, 12, 2048);
    // test_gelu(1, 2, 2, 3);
    // test_gelu(1, 1, 576, 2048);
    // test_conv_2d(1, 3, 5, 5, 3, 3, 2, 1, 0, 0, 1);
    // test_conv_2d(1, 2048, 24, 24, 1, 1, 2048, 1, 0, 0, 1);
    // test_conv_2d(1, 2048, 12, 12, 1, 1, 2048, 1, 0, 0, 1);
    // test_ldp();
    test_peg();

    return 0;
}
