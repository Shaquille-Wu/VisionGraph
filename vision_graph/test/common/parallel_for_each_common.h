/*
    Copyright (c) 2005-2020 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#ifndef __TBB_test_common_parallel_for_each_common_H
#define __TBB_test_common_parallel_for_each_common_H

#include "tbb/parallel_for_each.h"
#include "tbb/global_control.h"

#include "test.h"
#include "config.h"
#include "utils.h"
#include "utils_report.h"
#include "utils_concurrency_limit.h"
#include "iterator.h"
#include "cpu_usertime.h"

#include <vector>
#include <forward_list>

constexpr std::size_t depths_nubmer = 20;
static std::atomic<int> g_values_counter;

class value_t {
    size_t x;
    value_t& operator=(const value_t&);
public:
    value_t(size_t xx) : x(xx) { ++g_values_counter; }
    value_t(const value_t& v) : x(v.x) { ++g_values_counter; }
    value_t(value_t&& v) : x(v.x) { ++g_values_counter; }
    ~value_t() { --g_values_counter; }
    size_t value() const volatile { return x; }
};

static size_t g_tasks_expected = 0;
static std::atomic<size_t> g_tasks_observed;

size_t FindNumOfTasks(size_t max_depth) {
    if( max_depth == 0 )
        return 1;
    return  max_depth * FindNumOfTasks( max_depth - 1 ) + 1;
}

//! Simplest form of the parallel_for_each functor object.
struct FakeTaskGeneratorBody {
    //! The simplest form of the function call operator
    /** It does not allow adding new tasks during its execution. **/
    void operator()(value_t depth) const {
        g_tasks_observed += FindNumOfTasks(depth.value());
    }
};

/** Work item is passed by reference here. **/
struct FakeTaskGeneratorBody_RefVersion {
    void operator()(value_t& depth) const {
        g_tasks_observed += FindNumOfTasks(depth.value());
    }
};

/** Work item is passed by reference to const here. **/
struct FakeTaskGeneratorBody_ConstRefVersion {
    void operator()(const value_t& depth) const {
        g_tasks_observed += FindNumOfTasks(depth.value());
    }
};

/** Work item is passed by reference to volatile here. **/
struct FakeTaskGeneratorBody_VolatileRefVersion {
    void operator()(volatile value_t& depth, tbb::feeder<value_t>&) const {
        g_tasks_observed += FindNumOfTasks(depth.value());
    }
};

/** Work item is passed by rvalue reference here. **/
struct FakeTaskGeneratorBody_RvalueRefVersion {
    void operator()(value_t&& depth ) const {
        g_tasks_observed += FindNumOfTasks(depth.value());
    }
};

void do_work(const value_t& depth, tbb::feeder<value_t>& feeder) {
    ++g_tasks_observed;
    value_t new_value(depth.value()-1);
    for(size_t i = 0; i < depth.value(); ++i) {
        if (i%2) feeder.add( new_value );                // pass lvalue
        else     feeder.add( value_t(depth.value()-1) ); // pass rvalue
    }
}

//! Standard form of the parallel_for_each functor object.
/** Allows adding new work items on the fly. **/
struct TaskGeneratorBody {
    //! This form of the function call operator can be used when the body needs to add more work during the processing
    void operator()(value_t depth, tbb::feeder<value_t>& feeder) const {
        do_work(depth, feeder);
    }
private:
    // Assert that parallel_for_each does not ever access body constructors
    TaskGeneratorBody () {}
    TaskGeneratorBody (const TaskGeneratorBody&);
    // These functions need access to the default constructor
    template<class Body, class Iterator> friend void TestBody(size_t);
    template<class Body, class Iterator> friend void TestBody_MoveOnly(size_t);
};

/** Work item is passed by reference here. **/
struct TaskGeneratorBody_RefVersion {
    void operator()(value_t& depth, tbb::feeder<value_t>& feeder) const {
        do_work(depth, feeder);
    }
};

/** Work item is passed as const here. Compilers must ignore the const qualifier. **/
struct TaskGeneratorBody_ConstVersion {
    void operator()(const value_t depth, tbb::feeder<value_t>& feeder) const {
        do_work(depth, feeder);
    }
};

/** Work item is passed by reference to const here. **/
struct TaskGeneratorBody_ConstRefVersion {
    void operator()(const value_t& depth, tbb::feeder<value_t>& feeder) const {
        do_work(depth, feeder);
    }
};

/** Work item is passed by reference to volatile here. **/
struct TaskGeneratorBody_VolatileRefVersion {
    void operator()(volatile value_t& depth, tbb::feeder<value_t>& feeder) const {
        do_work(const_cast<value_t&>(depth), feeder);
    }
};

/** Work item is passed by reference to const volatile here. **/
struct TaskGeneratorBody_ConstVolatileRefVersion {
    void operator()(const volatile value_t& depth, tbb::feeder<value_t>& feeder) const {
        do_work(const_cast<value_t&>(depth), feeder);
    }
};

/** Work item is passed by rvalue reference here. **/
struct TaskGeneratorBody_RvalueRefVersion {
    void operator()(value_t&& depth, tbb::feeder<value_t>& feeder) const {
        do_work(depth, feeder);
    }
};

static value_t g_depths[depths_nubmer] = {0, 1, 2, 3, 4, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2};

template<class Body, class Iterator>
void TestBody_MoveIter(const Body& body, Iterator begin, Iterator end) {
    typedef std::move_iterator<Iterator> MoveIterator;
    MoveIterator mbegin(begin);
    MoveIterator mend(end);
    g_tasks_observed = 0;
    tbb::parallel_for_each(mbegin, mend, body);
    REQUIRE (g_tasks_observed == g_tasks_expected);
}

template<class Body, class Iterator>
void TestBody_MoveOnly(size_t depth) {
    typedef typename std::iterator_traits<Iterator>::value_type value_type;
    value_type a_depths[depths_nubmer] = {0, 1, 2, 3, 4, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2};
    TestBody_MoveIter(Body(), Iterator(a_depths), Iterator(a_depths + depth));
}

template<class Body, class Iterator>
void TestBody(size_t depth) {
    typedef typename std::iterator_traits<Iterator>::value_type value_type;
    value_type a_depths[depths_nubmer] = {0, 1, 2, 3, 4, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2};
    Body body;
    Iterator begin(a_depths);
    Iterator end(a_depths + depth);
    g_tasks_observed = 0;
    tbb::parallel_for_each(begin, end, body);
    REQUIRE (g_tasks_observed == g_tasks_expected);
    TestBody_MoveIter(body, Iterator(a_depths), Iterator(a_depths + depth));
}

template<class Iterator>
void TestIterator_Common(size_t depth) {
    TestBody<FakeTaskGeneratorBody, Iterator>(depth);
    TestBody<FakeTaskGeneratorBody_ConstRefVersion, Iterator>(depth);
    TestBody<TaskGeneratorBody, Iterator>(depth);
    TestBody<TaskGeneratorBody_ConstVersion, Iterator>(depth);
    TestBody<TaskGeneratorBody_ConstRefVersion, Iterator>(depth);
}

template<class Iterator>
void TestIterator_Const(size_t depth) {
    TestIterator_Common<Iterator>(depth);
    TestBody<TaskGeneratorBody_ConstVolatileRefVersion, Iterator>(depth);
}

#if __TBB_CPP14_GENERIC_LAMBDAS_PRESENT
template<class Iterator, class GenericBody>
void TestGenericLambda(size_t depth, GenericBody body) {
    typedef typename std::iterator_traits<Iterator>::value_type value_type;
    value_type a_depths[depths_nubmer] = {0, 1, 2, 3, 4, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2};
    Iterator begin(a_depths);
    Iterator end(a_depths + depth);
    g_tasks_observed = 0;
    tbb::parallel_for_each(begin, end, body);
    REQUIRE (g_tasks_observed == g_tasks_expected);
    TestBody_MoveIter(body, Iterator(a_depths), Iterator(a_depths + depth));
}

template<class Iterator>
void TestGenericLambdasCommon(size_t depth) {
    TestGenericLambda<Iterator>(depth, [](auto item){g_tasks_observed += FindNumOfTasks(item.value());});
    TestGenericLambda<Iterator>(depth, [](const auto item){g_tasks_observed += FindNumOfTasks(item.value());});
    TestGenericLambda<Iterator>(depth, [](volatile auto& item){g_tasks_observed += FindNumOfTasks(item.value());});
    TestGenericLambda<Iterator>(depth, [](const volatile auto& item){g_tasks_observed += FindNumOfTasks(item.value());});
    TestGenericLambda<Iterator>(depth, [](auto& item){g_tasks_observed += FindNumOfTasks(item.value());});
    TestGenericLambda<Iterator>(depth, [](const auto& item){g_tasks_observed += FindNumOfTasks(item.value());});
    TestGenericLambda<Iterator>(depth, [](auto&& item){g_tasks_observed += FindNumOfTasks(item.value());});

    TestGenericLambda<Iterator>(depth, [](auto item, auto& feeder){do_work(item, feeder);});
    TestGenericLambda<Iterator>(depth, [](const auto item, auto& feeder){do_work(item, feeder);});
    TestGenericLambda<Iterator>(depth, [](volatile auto& item, auto& feeder){do_work(const_cast<value_t&>(item), feeder);});
    TestGenericLambda<Iterator>(depth, [](const volatile auto& item, auto& feeder){do_work(const_cast<value_t&>(item), feeder);});
    TestGenericLambda<Iterator>(depth, [](auto& item, auto& feeder){do_work(item, feeder);});
    TestGenericLambda<Iterator>(depth, [](const auto& item, auto& feeder){do_work(item, feeder);});
    TestGenericLambda<Iterator>(depth, [](auto&& item, auto& feeder){do_work(item, feeder);});
}
#endif /*__TBB_CPP14_GENERIC_LAMBDAS_PRESENT*/

template<class Iterator>
void TestIterator_Modifiable(size_t depth) {
    TestIterator_Const<Iterator>(depth);
    TestBody<FakeTaskGeneratorBody_RefVersion, Iterator>(depth);
    TestBody<FakeTaskGeneratorBody_VolatileRefVersion, Iterator>(depth);
    TestBody<TaskGeneratorBody_RefVersion, Iterator>(depth);
    TestBody<TaskGeneratorBody_VolatileRefVersion, Iterator>(depth);
    TestBody_MoveOnly<FakeTaskGeneratorBody_RvalueRefVersion, Iterator>(depth);
    TestBody_MoveOnly<TaskGeneratorBody_RvalueRefVersion, Iterator>(depth);
#if __TBB_CPP14_GENERIC_LAMBDAS_PRESENT
    TestGenericLambdasCommon<Iterator>(depth);
#endif
}

std::atomic<std::size_t> task_counter{0};

template<template <typename> class IteratorModifier>
struct generic_iterator_container {
    using value_type = std::size_t;
    using container_type = std::vector<value_type>;
    using iterator_type = IteratorModifier<value_type>;

    static constexpr std::size_t default_size = 100;
    container_type my_vec;
    iterator_type my_begin, my_end;

    generic_iterator_container():
        my_vec   (default_size),
        my_begin {my_vec.data()}, 
        my_end   {my_vec.data() + my_vec.size()}
    {}

    iterator_type begin() const { return my_begin; }
    iterator_type end()   const { return my_end;   }

    void validation(std::size_t expected_value) {
        for (std::size_t i = 0; i < my_vec.size(); i++)
            REQUIRE_MESSAGE(my_vec[i] == expected_value, "Some element was not produced");

        REQUIRE_MESSAGE(task_counter == my_vec.size(), "Not all elements were produced");
        task_counter = 0;
    }
};

struct incremental_functor {
    void operator()(std::size_t& in) const { ++in; ++task_counter;}
};

template< template<typename> class IteratorModifier>
void container_based_overload_test_case(std::size_t expected_value) {
    generic_iterator_container<IteratorModifier> container;
    tbb::parallel_for_each(container, incremental_functor{});
    container.validation(expected_value);
}

namespace TestMoveSem {
    struct MovePreferable : utils::Movable {
        MovePreferable() : Movable(), addtofeed(true) {}
        MovePreferable(bool addtofeed_) : Movable(), addtofeed(addtofeed_) {}
        MovePreferable(MovePreferable&& other) : Movable(std::move(other)),
                                                 addtofeed(other.addtofeed) {};
        // base class is explicitly initialized in the copy ctor to avoid -Wextra warnings
        MovePreferable(const MovePreferable& other) : Movable(other) {
            REPORT("Error: copy ctor preferred.\n");
        };
        MovePreferable& operator=(const MovePreferable&) {
            REPORT("Error: copy assign operator preferred.\n"); return *this;
        }
        bool addtofeed;
    };
    struct MoveOnly : MovePreferable {
        MoveOnly( const MoveOnly& ) = delete;
        MoveOnly() : MovePreferable() {}
        MoveOnly(bool addtofeed_) : MovePreferable(addtofeed_) {}
        MoveOnly(MoveOnly&& other) : MovePreferable(std::move(other)) {};
    };
}

template<typename T>
void RecordAndAdd(const T& in, tbb::feeder<T>& feeder) {
    REQUIRE_MESSAGE(in.alive, "Got dead object in body");
    size_t i = ++g_tasks_observed;
    if (in.addtofeed) {
        if (i%2) feeder.add(T(false));
        else {
            T a(false);
            feeder.add(std::move(a));
        }
    }
}

// Take an item by rvalue reference
template<typename T>
struct TestMoveIteratorBody {
    void operator() (T&& in, tbb::feeder<T>& feeder) const { RecordAndAdd(in, feeder); }
};

// Take an item by value
template<typename T>
struct TestMoveIteratorBodyByValue {
    void operator() (T in, tbb::feeder<T>& feeder) const { RecordAndAdd(in, feeder); }
};

template<typename Iterator, typename Body>
void TestMoveIterator() {
    typedef typename std::iterator_traits<Iterator>::value_type value_type;

    Body body;
    const size_t size = 65;
    g_tasks_observed = 0;
    value_type a[size];
    tbb::parallel_for_each(std::make_move_iterator(Iterator(a)), std::make_move_iterator(Iterator(a+size)), body);
    REQUIRE(size * 2  == g_tasks_observed);
}

template<typename T>
void DoTestMoveSemantics() {
    TestMoveIterator<utils::InputIterator<T>, TestMoveIteratorBody<T>>();
    TestMoveIterator<utils::ForwardIterator<T>, TestMoveIteratorBody<T>>();
    TestMoveIterator<utils::RandomIterator<T>, TestMoveIteratorBody<T>>();

    TestMoveIterator<utils::InputIterator<T>, TestMoveIteratorBodyByValue<T>>();
    TestMoveIterator<utils::ForwardIterator<T>, TestMoveIteratorBodyByValue<T>>();
    TestMoveIterator<utils::RandomIterator<T>, TestMoveIteratorBodyByValue<T>>();
}

#endif // __TBB_test_common_parallel_for_each_common_H
