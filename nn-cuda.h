#ifndef NN_H_
#define NN_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

// #define NN_BACKPROP_TRADITIONAL

#ifndef NN_ACT
#define NN_ACT ACT_SIG
#endif // NN_ACT

#ifndef NN_RELU_PARAM
#define NN_RELU_PARAM 0.01f
#endif // NN_RELU_PARAM

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])

typedef enum {
    ACT_SIG,
    ACT_RELU,
    ACT_TANH,
    ACT_SIN,
} Act;

float rand_float(void);

float sigmoidf(float x);
float reluf(float x);
float tanhf(float x);

// Dispatch to the corresponding activation function
float actf(float x, Act act);

// Derivative of the activation function based on its value
float dactf(float y, Act act);

typedef struct {
    size_t capacity;
    size_t size;
    uintptr_t *words;
} Region;

// capacity is in bytes, but it can allocate more just to keep things
// word aligned
Region region_alloc_alloc(size_t capacity_bytes);
void *region_alloc(Region *r, size_t size_bytes);
#define region_reset(r) (NN_ASSERT((r) != NULL), (r)->size = 0, cudaFree((r)->words))
#define region_free(r) (region_reset(r), cudaFree((r)->words))
#define region_occupied_bytes(r) (NN_ASSERT((r) != NULL), (r)->size*sizeof(*(r)->words))
#define region_save(r) (NN_ASSERT((r) != NULL), (r)->size)
#define region_rewind(r, s) (NN_ASSERT((r) != NULL), (r)->size = s)

typedef struct {
    size_t rows;
    size_t cols;
    float *elements;
} Mat;

typedef struct {
    size_t cols;
    float *elements;
} Row;

#define ROW_AT(row, col) (row).elements[col]

Mat row_as_mat(Row row);
#define row_alloc(r, cols) mat_row(mat_alloc(r, 1, cols), 0)
Row row_slice(Row row, size_t i, size_t cols);
#define row_rand(row, low, high) mat_rand(row_as_mat(row), low, high)
#define row_fill(row, x) mat_fill(row_as_mat(row), x);
#define row_print(row, name, padding) mat_print(row_as_mat(row), name, padding)
#define row_copy(dst, src) mat_copy(row_as_mat(dst), row_as_mat(src))

#define MAT_AT(m, i, j) (m).elements[(i)*(m).cols + (j)]

Mat mat_alloc(Region *r, size_t rows, size_t cols);
void mat_fill(Mat m, float x);
void mat_rand(Mat m, float low, float high);
Row mat_row(Mat m, size_t row);
void mat_copy(Mat dst, Mat src);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_sum(Mat dst, Mat a);
void mat_act(Mat m);
void mat_print(Mat m, const char *name, size_t padding);
void mat_shuffle_rows(Mat m);
#define MAT_PRINT(m) mat_print(m, #m, 0)

typedef struct {
    size_t *arch;
    size_t arch_count;
    Mat *ws; // The amount of activations is arch_count-1
    Row *bs; // The amount of activations is arch_count-1

    // TODO: maybe remove these? It would be better to allocate them in a
    // temporary region during the actual forwarding
    Row *as;
} NN;

#define NN_INPUT(nn) (NN_ASSERT((nn).arch_count > 0), (nn).as[0])
#define NN_OUTPUT(nn) (NN_ASSERT((nn).arch_count > 0), (nn).as[(nn).arch_count-1])

NN nn_alloc(Region *r, size_t *arch, size_t arch_count);
void nn_zero(NN nn);
void nn_print(NN nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn);
void nn_rand(NN nn, float low, float high);
// TODO: make nn_forward signature more natural
//
// Something more like `Mat nn_forward(NN nn, Mat in)`
void nn_forward(NN nn);
float nn_cost(NN nn, Mat t);
NN nn_finite_diff(Region *r, NN nn, Mat t, float eps);
NN nn_backprop(Region *r, NN nn, Mat t);
void nn_learn(NN nn, NN g, float rate);

typedef struct {
    size_t begin;
    float cost;
    bool finished;
} Batch;

void batch_process(Region *r, Batch *b, size_t batch_size, NN nn, Mat t, float rate);

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
__global__ void addKernel(int* c, const int* a, const int* b);

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

__device__ float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

__device__ float reluf(float x)
{
    return x > 0 ? x : x*NN_RELU_PARAM;
}

__device__ float tanhf(float x)
{
    float ex = expf(x);
    float enx = expf(-x);
    return (ex - enx)/(ex + enx);
}

__device__ float actf(float x, Act act)
{
    switch (act) {
    case ACT_SIG:  return sigmoidf(x);
    case ACT_RELU: return reluf(x);
    case ACT_TANH: return tanhf(x);
    case ACT_SIN:  return sinf(x);
    }
    NN_ASSERT(0 && "Unreachable");
    return 0.0f;
}

__device__ float dactf(float y, Act act)
{
    switch (act) {
    case ACT_SIG:  return y*(1 - y);
    case ACT_RELU: return y >= 0 ? 1 : NN_RELU_PARAM;
    case ACT_TANH: return 1 - y*y;
    case ACT_SIN:  return cosf(asinf(y));
    }
    NN_ASSERT(0 && "Unreachable");
    return 0.0f;
}

Mat mat_alloc(Region *r, size_t rows, size_t cols)
{
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.elements = (float*) region_alloc(r, sizeof(*m.elements)*rows*cols);
    NN_ASSERT(m.elements != NULL);
    return m;
}

NN nn_alloc(Region *r, size_t *arch, size_t arch_count)
{
    NN_ASSERT(arch_count > 0);

    NN nn;
    nn.arch = arch;
    nn.arch_count = arch_count;

    nn.ws = (Mat*) region_alloc(r, sizeof(*nn.ws)*(nn.arch_count - 1));
    NN_ASSERT(nn.ws != NULL);
    nn.bs = (Row*) region_alloc(r, sizeof(*nn.bs)*(nn.arch_count - 1));
    NN_ASSERT(nn.bs != NULL);
    nn.as = (Row*) region_alloc(r, sizeof(*nn.as)*nn.arch_count);
    NN_ASSERT(nn.as != NULL);

    nn.as[0] = row_alloc(r, arch[0]);
    for (size_t i = 1; i < arch_count; ++i) {
        nn.ws[i-1] = mat_alloc(r, nn.as[i-1].cols, arch[i]);
        nn.bs[i-1] = row_alloc(r, arch[i]);
        nn.as[i]   = row_alloc(r, arch[i]);
    }

    return nn;
}

Region region_alloc_alloc(size_t capacity_bytes)
{
    Region r = {0};

    size_t word_size = sizeof(*r.words);
    size_t capacity_words = (capacity_bytes + word_size - 1)/word_size;

    uintptr_t *words;
    cudaMalloc((void**)&words, capacity_words*word_size);
    NN_ASSERT(words != NULL);
    r.capacity = capacity_words;
    r.words = words;
    return r;
}

void *region_alloc(Region *r, size_t size_bytes)
{
    if (r == NULL) return NN_MALLOC(size_bytes);
    size_t word_size = sizeof(*r->words);
    size_t size_words = (size_bytes + word_size - 1)/word_size;

    NN_ASSERT(r->size + size_words <= r->capacity);
    if (r->size + size_words > r->capacity) return NULL;
    void *result = &r->words[r->size];
    r->size += size_words;
    return result;
}

void* region_alloc_memcpy(Region *r, size_t size_bytes, void* data)
{
    if (r == NULL) return NN_MALLOC(size_bytes);
    size_t word_size = sizeof(*r->words);
    size_t size_words = (size_bytes + word_size - 1)/word_size;

    NN_ASSERT(r->size + size_words <= r->capacity);
    if (r->size + size_words > r->capacity) return NULL;
    void *result = &r->words[r->size];
    r->size += size_words;
    cudaMemcpy(result, data, size_bytes, cudaMemcpyHostToDevice);
    return result;
}

Row mat_row(Mat m, size_t row)
{
    return /*(Row)*/ {
        /*.cols     =   */ m.cols,
        /*.elements =   */ &MAT_AT(m, row, 0),
    };
}


__global__ void addKernel(uintptr_t *c, const uintptr_t *a, const uintptr_t *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(uintptr_t *c, const uintptr_t *a, const uintptr_t *b, unsigned int size)
{
    cudaError_t cudaStatus;

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(c, a, b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        return cudaStatus; 
    }

    
    return cudaStatus;
}
#endif  // NN_IMPLEMENTATION
