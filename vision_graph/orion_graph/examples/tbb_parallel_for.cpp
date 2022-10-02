#include "tbb/partitioner.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include <time.h>
#include <math.h>
#include <chrono> 

using namespace tbb;

static void Foo(float* src, float* dst, int data_count)
{
    for(int i = 0 ; i < data_count ; i ++)
        dst[i] = sqrtf(src[i]);
}

class ApplyFoo
{
    float* src_;
    float* dst_;
    int    data_count_;
    public:
    void operator () (const blocked_range<size_t> & r) const
    {
        for (size_t i = r.begin(); i != r.end(); ++ i)
        {
            float* src = src_ + i * data_count_;
            float* dst = dst_ + i * data_count_;
            Foo(src, dst, data_count_);
        }
    }

    ApplyFoo(float* src, float* dst, int data_count) : src_(src), dst_(dst), data_count_(data_count) {;};
};

int GenerateValue(float* data, int data_count)
{
    for(int i = 0 ; i < data_count ; i ++)
        data[i] = (rand() % 1000);

    return 0;
}

int main(int argc, char* argv[])
{
    // 创建task scheduler
    srand(time(0));
    //task_scheduler_init init;
    static const int task_count = 16;
    static const int data_count = 1024*1024;
    float*           src_data   = new float[task_count * data_count];
    float*           dst_data   = new float[task_count * data_count];
    GenerateValue(src_data, task_count * data_count);
    ApplyFoo         app(src_data, dst_data, data_count);

    auto start    = std::chrono::system_clock::now();
    for(int i = 0 ; i < 256 ; i ++)
        parallel_for(blocked_range<size_t>(0, task_count), app, auto_partitioner());
    auto end      = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("tbb parallel_for cost %lld\n", (long long int)(duration.count()));

    start    = std::chrono::system_clock::now();
    for(int i = 0 ; i < 256 ; i ++)
        Foo(src_data, dst_data, task_count * data_count);
    end      = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("without tbb parallel_for cost %lld\n", (long long int)(duration.count()));

    delete[] src_data;
    delete[] dst_data;
    return 0;
}