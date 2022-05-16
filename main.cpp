# include <iostream>
# include <omp.h>
# include <cmath>
# include <ctime>
# include <algorithm>
# include <chrono>
# include <stdlib.h>
#include <arm_neon.h>


# define START
//# define PRINT
# define MP_START

using namespace std;

struct Timer
{
    /* data */
    using clock_t = std::chrono::high_resolution_clock;
    using time_point_t = std::chrono::time_point<clock_t>;
    using elapsed_time_t = std::chrono::duration<double, std::milli>;

    time_point_t StartTime;
    time_point_t StopTime;
    elapsed_time_t ElapsedTime;

    void Start()
    {
        ElapsedTime = elapsed_time_t::zero();
        StartTime = clock_t::now();
    }

    void Stop()
    {
        StopTime = clock_t::now();
        std::chrono::duration<double, std::milli> elapsedTime = StopTime - StartTime;
        std::cout << elapsedTime.count();
    }
};

void replace_neon(float* input, float* output, int count)
{
    float32x4_t d = vmovq_n_f32(0.0);
    int new_count = count;
    if (count % 4 != 0)
    {
        new_count = 4 * (count / 4);
    }
    for (int i = 0; i < new_count; i += 4)
    {
        float32x4_t in, out;
        in = vld1q_f32(input);
        out = vaddq_f32(in, d);
        vst1q_f32(output, out);
        input += 4;
    }
    for (int i = 0; i < count - new_count; i++)
    {
        output[i] = input[i];
    }
}

void init_matrix(float* m, float* b, int size)
{
    srand(time(0));
    float sum = 0;
        for (int i = 0; i < size; i++)
        {
            sum = 0;
            for (int j = 0; j < size; j++)
            {
                if (j == i)
                    continue;;
                m[i * size + j] = rand() % 10 - 5;
                sum += abs(m[i * size + j]);
            }
            if (rand() & 1)
                m[i * size + i] = sum + 1;
            else
                m[i * size + i] = -sum - 1;
        }
        for (int i = 0; i < size; i++)
        {
            b[i] = rand() % 10 - 5;
        }
}

//jacobi算法
int main()
{
    const int size = 10000;
    float* x = new float[size];
    float* temp = new float[size];
    float* b = new float[size];
    float* Matrix = new float[size * size];
    int max_iter = 200;
    float esp = 1e-10;
    float residual = 0.0;
    int iter = 0;
    init_matrix(Matrix, b, size);
    Timer T;

# ifdef PRINT
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            cout << Matrix[i * size + j] << "  ";
        }
        cout << b[i] << endl;
    }
# endif

# ifdef START

    //初始解设置为全0
    for (int i = 0; i < size; i++)
    {
        x[i] = 0.0;
    }

    iter = 0;
    T.Start();
    while (iter < max_iter)
    {
        float d = 0.0;
        for (int i = 0; i < size; i++)
        {
            float v = b[i];
            for (int j = 0; j < size; j++)
            {
                if (j != i)
                {
                    v -= Matrix[j + i * size] * x[j];
                }
            }
            v /= Matrix[i + i * size];
            temp[i] = v;
        }
        for (int k = 0; k < size; k++)
        {
            d += abs(x[k] - temp[k]);
        }
        //更新x的值
        for (int k = 0; k < size; k++)
        {
            x[k] = temp[k];
        }
        iter++;
        if (d < esp)
            break;
    }
    cout << "Normal time = ";
    T.Stop();
    cout << " ms" << endl;
    cout << "迭代次数：" << iter << endl;
    //验证结果
    residual = 0.0;
    for (int i = 0; i < size; i++)
    {
        float v = b[i];
        for (int j = 0; j < size; j++)
        {
            v -= Matrix[i * size + j] * x[j];
        }
        residual += abs(v);
    }
    cout << "residual = " << residual << endl;

# endif

# ifdef MP_START

    //初始解设置为全0
    for (int i = 0; i < size; i++)
    {
        x[i] = 0.0;
    }

    iter = 0;
    omp_set_num_threads(8);

    T.Start();
    while (iter < max_iter)
    {
        float d = 0.0;
        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < size; i++)
            {
                float v = b[i];
                for (int j = 0; j < size; j++)
                {
                    if (j != i)
                    {
                        v -= Matrix[j + i * size] * x[j];
                    }
                }
                v /= Matrix[i + i * size];
                temp[i] = v;
            }
        }
        for (int k = 0; k < size; k++)
        {
            d += abs(x[k] - temp[k]);
        }
        //更新x的值
        //用SIMD并行
        replace_neon(x, temp, size);
        //for (int k = 0; k < size; k++)
        //{
        //    x[k] = temp[k];
        //}
        iter++;
        if (d < esp)
            break;
    }
    cout << "openMP computing time = ";
    T.Stop();
    cout << " ms" << endl;
    cout << "迭代次数：" << iter << endl;

    //验证结果
    residual = 0.0;
    for (int i = 0; i < size; i++)
    {
        float v = b[i];
        for (int j = 0; j < size; j++)
        {
            v -= Matrix[i * size + j] * x[j];
        }
        residual += abs(v);
    }
    cout << "residual = " << residual << endl;

# endif // 


    return 0;
}