/*
 * Copyright (C) OrionStart Technology(Beijing) Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file parallel_task.h
 * @brief This header file defines parallel execution.
 * @author WuXiao(wuxiao@ainirobot.com)
 * @date 2020-07-06
 */

#ifndef PARALLEL_TASK_H_
#define PARALLEL_TASK_H_

#include <vector>
#include "tbb/blocked_range.h"
#include "tbb/parallel_for_each.h"
#include "tbb/partitioner.h"

namespace vision_graph
{

/**
 * @brief The Invoker object is the base class for 'task_unit'.
 * 
 */
template <typename TASK> struct TaskInvoker {
    void operator()(TASK& exe) const 
    {
        exe.Run();
    }
};

/**
 * @brief execute tasks with parallel style
 * 
 * @param tasks the executable object
 * 
 * @param task_count how many task count you want to partition into
 * 
 * @return none
 * 
 */
template<typename TASK>
inline void ParallelRun(std::vector<TASK>& tasks) noexcept
{
    tbb::parallel_for_each(tasks.begin(),tasks.end(), TaskInvoker<TASK>());
};

} //namespace vision_graph
#endif