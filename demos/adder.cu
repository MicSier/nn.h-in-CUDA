#define GYM_IMPLEMENTATION
#include "../gym.h"

#define NN_IMPLEMENTATION
#include "../nn-cuda.h"

#define BITS 5

size_t arch[] = {2*BITS, 4*BITS, BITS + 1};
size_t epoch = 0;
size_t max_epoch = 100*1000;
size_t batches_per_frame = 200;
size_t batch_size = 28;
float rate = 1.0f;
bool paused = true;

/*void verify_nn_adder(Font font, NN nn, Gym_Rect r)
{
    float s;
    if (r.w < r.h) {
        s = r.w - r.w*0.05;
        r.y = r.y + r.h/2 - s/2;
    } else {
        s = r.h - r.h*0.05;
        r.x = r.x + r.w/2 - s/2;
    }
    size_t n = 1<<BITS;
    float cs = s/n;

    for (size_t x = 0; x < n; ++x) {
        for (size_t y = 0; y < n; ++y) {
            for (size_t i = 0; i < BITS; ++i) {
                ROW_AT(NN_INPUT(nn), i)        = (x>>i)&1;
                ROW_AT(NN_INPUT(nn), i + BITS) = (y>>i)&1;
            }

            nn_forward(nn);

            size_t z = 0.0f;
            for (size_t i = 0; i < BITS; ++i) {
                size_t bit = ROW_AT(NN_OUTPUT(nn), i) > 0.5;
                z = z|(bit<<i);
            }
            bool overflow = ROW_AT(NN_OUTPUT(nn), BITS) > 0.5;
            bool correct = z == x + y;

            Vector2 position = { r.x + x*cs, r.y + y*cs };
            Vector2 size = { cs, cs };

            if (correct)  DrawRectangleV(position, size, DARKGREEN);
            if (overflow) DrawRectangleV(position, size, DARKPURPLE);

            char buffer[256];
            snprintf(buffer, sizeof(buffer), "%zu", z);

            // Centering the text
            float fontSize = cs*0.8;
            float spacing = 0;
            Vector2 text_size = MeasureTextEx(font, buffer, fontSize, spacing);
            position.x = position.x + cs/2 - text_size.x/2;
            position.y = position.y + cs/2 - text_size.y/2;

            DrawTextEx(font, buffer, position, fontSize, spacing, WHITE);
        }
    }
}*/
/*
int main(void)
{
    Region temp = region_alloc_alloc(256*1024*1024);

    size_t n = (1<<BITS);
    size_t rows = n*n;
    Mat t  = mat_alloc(NULL, rows, 2*BITS + BITS + 1);
    for (size_t i = 0; i < t.rows; ++i) {
        Row row = mat_row(t, i);
        Row in = row_slice(row, 0, 2*BITS);
        Row out = row_slice(row, in.cols, BITS + 1);
        size_t x = i/n;
        size_t y = i%n;
        size_t z = x + y;
        for (size_t j = 0; j < BITS; ++j) {
            ROW_AT(in, j)        = (x>>j)&1;
            ROW_AT(in, j + BITS) = (y>>j)&1;
            ROW_AT(out, j)       = (z>>j)&1;
        }
        if (z >= n) {
            for (size_t j = 0; j < BITS; ++j) {
                ROW_AT(out, j) = 1;
            }
            ROW_AT(out, BITS) = 1;
        } else {
            ROW_AT(out, BITS) = 0;
        }
    }

    NN nn = nn_alloc(NULL, arch, ARRAY_LEN(arch));
    nn_rand(nn, -1, 1);

    size_t WINDOW_FACTOR = 80;
    size_t WINDOW_WIDTH = (16*WINDOW_FACTOR);
    size_t WINDOW_HEIGHT = (9*WINDOW_FACTOR);

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "xor");
    SetTargetFPS(60);

    Font font = LoadFontEx("./fonts/iosevka-regular.ttf", 72, NULL, 0);
    SetTextureFilter(font.texture, TEXTURE_FILTER_BILINEAR);

    Gym_Plot plot = {0};
    Batch batch = {0};

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) {
            paused = !paused;
        }
        if (IsKeyPressed(KEY_R)) {
            epoch = 0;
            nn_rand(nn, -1, 1);
            plot.count = 0;
        }

        for (size_t i = 0; i < batches_per_frame && !paused && epoch < max_epoch; ++i) {
            batch_process(&temp, &batch, batch_size, nn, t, rate);
            if (batch.finished) {
                epoch += 1;
                da_append(&plot, batch.cost);
                mat_shuffle_rows(t);
            }
        }

        BeginDrawing();
        ClearBackground(GYM_BACKGROUND);
        {
            int w = GetRenderWidth();
            int h = GetRenderHeight();

            Gym_Rect r;
            r.w = w;
            r.h = h*2/3;
            r.x = 0;
            r.y = h/2 - r.h/2;

            gym_layout_begin(GLO_HORZ, r, 3, 10);
                gym_plot(plot, gym_layout_slot(), RED);
                gym_layout_begin(GLO_VERT, gym_layout_slot(), 2, 0);
                    gym_render_nn(nn, gym_layout_slot());
                    gym_render_nn_weights_heatmap(nn, gym_layout_slot());
                gym_layout_end();
                verify_nn_adder(font, nn, gym_layout_slot());
            gym_layout_end();

            char buffer[256];
            snprintf(buffer, sizeof(buffer), "Epoch: %zu/%zu, Rate: %f, Cost: %f, Temporary Memory: %zu\n", epoch, max_epoch, rate, nn_cost(nn, t), region_occupied_bytes(&temp));
            DrawTextEx(font, buffer, CLITERAL(Vector2){}, h*0.04, 0, WHITE);
        }
        EndDrawing();

        region_reset(&temp);
    }


    return 0;
}*/

void display_mat(Mat a)
{
    printf("---------------------------\n");
    for (int i = 0; i < a.cols; i++)
    {
        for (int j = 0; j < a.rows; j++)
        {
            printf("(%d,%d) = %f ",i,j,a.elements[a.cols * i + j]);
        }
        printf("\n");
    }
    printf("---------------------------\n");
}

Cuda_Config cuda_configure()
{
    int max_threads;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Assumes device 0
    cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerBlock, 0);
    Cuda_Config config = { max_threads };
    return config;
}

int main()
{
    Cuda_Config cuda_config = cuda_configure();
    Region temp = region_alloc_alloc(256*1024*1024);
    const int n = 1024;

    Mat a = mat_alloc(NULL,n,n);
    Mat b = mat_alloc(NULL,n,n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            a.elements[n * i + j] = i + j;
            b.elements[n * i + j] = i - j;
        }
    }    

    Mat dev_c =  mat_alloc(&temp, n , n);

    Mat dev_a =  mat_alloc_memcpy(&temp, n,n, a);
    Mat dev_b = mat_alloc_memcpy(&temp, n,n, b);

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(dev_c, dev_a, dev_b, n, cuda_config);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
    Mat c = mat_alloc_memcpy(NULL, n, n, dev_c);
    
    //display_mat(a);
    //display_mat(b);
    //display_mat(c);

    // Clean up
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    region_free(&temp);
    free(a.elements);
    free(b.elements);
    free(c.elements);

    return 0;
}


