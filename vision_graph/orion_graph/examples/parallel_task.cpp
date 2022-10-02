#include <time.h>
#include <math.h>
#include <chrono>
#include "../vision_graph/parallel_task.h"

using namespace tbb;

class SqrtTask
{
public:
    SqrtTask() noexcept : src_(0), dst_(0), data_count_(0) {;};
    SqrtTask(SqrtTask const& other) noexcept : src_(other.src_), dst_(other.dst_), data_count_(other.data_count_) {;};
    SqrtTask(float* src, float* dst, int data_count) noexcept : src_(src), dst_(dst), data_count_(data_count) {;};
    SqrtTask& operator=(SqrtTask const& other) noexcept
    {
        if(&other == this)  return *this;

        src_        = other.src_;
        dst_        = other.dst_;
        data_count_ = other.data_count_;

        return *this;
    };

    int Run() noexcept
    {
        SqrtFunc(src_, dst_, data_count_);
        return 0;
    };

    static void SqrtFunc(float* src, float* dst, int data_count)
    {
        for(int i = 0 ; i < data_count ; i ++)
            dst[i] = sqrtf(src[i]);
    };

protected:
    float*     src_;
    float*     dst_;
    int        data_count_;
};

int GenerateValue(float* data, int data_count)
{
    for(int i = 0 ; i < data_count ; i ++)
        data[i] = (rand() % 1000);

    return 0;
}

int main(int argc, char* argv[])
{
    srand(time(0));
    int              i          = 0;
    static const int task_count = 8;
    static const int data_count = 1024*1024;
    float*           src_data   = new float[task_count * data_count];
    float*           dst_data   = new float[task_count * data_count];
    GenerateValue(src_data, task_count * data_count);
    std::vector<SqrtTask>  tasks(task_count);
    for(i = 0 ; i < task_count ; i ++)
        tasks[i] = SqrtTask(src_data + i * data_count, dst_data + i * data_count, data_count);

    auto start    = std::chrono::system_clock::now();
    for(i = 0 ; i < 256 ; i ++)
        vision_graph::ParallelRun<SqrtTask>(tasks);
    auto end      = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("tbb parallel_for cost %lld\n", (long long int)(duration.count()));

    start    = std::chrono::system_clock::now();
    for(int i = 0 ; i < 256 ; i ++)
        SqrtTask::SqrtFunc(src_data, dst_data, task_count * data_count);
    end      = std::chrono::system_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("without tbb parallel_for cost %lld\n", (long long int)(duration.count()));

    delete[] src_data;
    delete[] dst_data;
    return 0;
}