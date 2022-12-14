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

#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "common/test.h"
#include "common/utils.h"
#include "common/utils_report.h"

#include "tbb/parallel_for.h"
#include "tbb/tick_count.h"

#include "../tbb/test_partitioner.h"

#include <atomic>

//! \file conformance_parallel_for.cpp
//! \brief Test for [algorithms.parallel_for algorithms.auto_partitioner algorithms.simple_partitioner algorithms.static_partitioner algorithms.affinity_partitioner] specification

static const int N = 500;
static std::atomic<int> Array[N];

struct parallel_tag {};
struct empty_partitioner_tag {};

// Testing parallel_for with step support
const std::size_t PFOR_BUFFER_TEST_SIZE = 1024;
// test_buffer has some extra items beyond its right bound
const std::size_t PFOR_BUFFER_ACTUAL_SIZE = PFOR_BUFFER_TEST_SIZE + 1024;
size_t pfor_buffer[PFOR_BUFFER_ACTUAL_SIZE];

template<typename T>
class TestFunctor{
public:
    void operator ()(T index) const {
        pfor_buffer[index]++;
    }
};

static std::atomic<int> FooBodyCount;

// A range object whose only public members are those required by the Range concept.
template<size_t Pad>
class FooRange {
    // Start of range
    int start;

    // Size of range
    int size;
    FooRange( int start_, int size_ ) : start(start_), size(size_) {
        utils::zero_fill<char>(pad, Pad);
        pad[Pad-1] = 'x';
    }
    template<typename Flavor_, std::size_t Pad_> friend void Flog( );
    template<size_t Pad_> friend class FooBody;
    void operator&();

    char pad[Pad];
public:
    bool empty() const {return size==0;}
    bool is_divisible() const {return size>1;}
    FooRange( FooRange& original, tbb::split ) : size(original.size/2) {
        original.size -= size;
        start = original.start+original.size;
        CHECK( original.pad[Pad-1]=='x');
        pad[Pad-1] = 'x';
    }
};

// A range object whose only public members are those required by the parallel_for.h body concept.
template<size_t Pad>
class FooBody {
public:
    ~FooBody() {
        --FooBodyCount;
        for( std::size_t i=0; i<sizeof(*this); ++i )
            reinterpret_cast<char*>(this)[i] = -1;
    }
    // Copy constructor
    FooBody( const FooBody& other ) : array(other.array), state(other.state) {
        ++FooBodyCount;
        CHECK(state == LIVE);
    }
    void operator()( FooRange<Pad>& r ) const {
        for( int k=0; k<r.size; ++k ) {
            const int i = array[r.start+k]++;
            CHECK( i==0 );
        }
    }
private:
    const int LIVE = 0x1234;
    std::atomic<int>* array;
    int state;
    friend class FooRange<Pad>;
    template<typename Flavor_, std::size_t Pad_> friend void Flog( );
    FooBody( std::atomic<int>* array_ ) : array(array_), state(LIVE) {}
};

template <typename Flavor, typename Partitioner, typename Range, typename Body>
struct Invoker;

template <typename Range, typename Body>
struct Invoker<parallel_tag, empty_partitioner_tag, Range, Body> {
    void operator()( const Range& r, const Body& body, empty_partitioner_tag& ) {
        tbb::parallel_for( r, body );
    }
};

template <typename Partitioner, typename Range, typename Body>
struct Invoker<parallel_tag, Partitioner, Range, Body> {
    void operator()( const Range& r, const Body& body, Partitioner& p ) {
        tbb::parallel_for( r, body, p );
    }
};

template <typename Flavor, typename Partitioner, typename T, typename Body>
struct InvokerStep;

template <typename T, typename Body>
struct InvokerStep<parallel_tag, empty_partitioner_tag, T, Body> {
    void operator()( const T& first, const T& last, const Body& f, empty_partitioner_tag& ) {
        tbb::parallel_for( first, last, f );
    }
    void operator()( const T& first, const T& last, const T& step, const Body& f, empty_partitioner_tag& ) {
        tbb::parallel_for( first, last, step, f );
    }
};

template <typename Partitioner, typename T, typename Body>
struct InvokerStep<parallel_tag, Partitioner, T, Body> {
    void operator()( const T& first, const T& last, const Body& f, Partitioner& p ) {
        tbb::parallel_for( first, last, f, p );
    }
    void operator()( const T& first, const T& last, const T& step, const Body& f, Partitioner& p ) {
        tbb::parallel_for( first, last, step, f, p );
    }
};

template<typename Flavor, std::size_t Pad>
void Flog() {
    for ( int i=0; i<N; ++i ) {
        for ( int mode = 0; mode < 4; ++mode) {
            FooRange<Pad> r( 0, i );
            const FooRange<Pad> rc = r;
            FooBody<Pad> f( Array );
            const FooBody<Pad> fc = f;
            for (int a_i = 0; a_i < N; a_i++) {
                Array[a_i].store(0, std::memory_order_relaxed);
            }
            FooBodyCount = 1;
            switch (mode) {
            case 0: {
                empty_partitioner_tag p;
                Invoker< Flavor, empty_partitioner_tag, FooRange<Pad>, FooBody<Pad> > invoke_for;
                invoke_for( rc, fc, p );
            }
                break;
            case 1: {
                Invoker< Flavor, const tbb::simple_partitioner, FooRange<Pad>, FooBody<Pad> > invoke_for;
                invoke_for( rc, fc, tbb::simple_partitioner() );
            }
                break;
            case 2: {
                Invoker< Flavor, const tbb::auto_partitioner, FooRange<Pad>, FooBody<Pad> > invoke_for;
                invoke_for( rc, fc, tbb::auto_partitioner() );
            }
                break;
            case 3: {
                static tbb::affinity_partitioner affinity;
                Invoker< Flavor, tbb::affinity_partitioner, FooRange<Pad>, FooBody<Pad> > invoke_for;
                invoke_for( rc, fc, affinity );
            }
                break;
            }
            for( int j=0; j<i; ++j )
                CHECK( Array[j]==1);
            for( int j=i; j<N; ++j )
                CHECK( Array[j]==0);
            CHECK( FooBodyCount==1);
        }
    }
}

#include <stdexcept> // std::invalid_argument

template <typename Flavor, typename T, typename Partitioner>
void TestParallelForWithStepSupportHelper(Partitioner& p) {
    const T pfor_buffer_test_size = static_cast<T>(PFOR_BUFFER_TEST_SIZE);
    const T pfor_buffer_actual_size = static_cast<T>(PFOR_BUFFER_ACTUAL_SIZE);
    // Testing parallel_for with different step values
    InvokerStep< Flavor, Partitioner, T, TestFunctor<T> > invoke_for;
    for (T begin = 0; begin < pfor_buffer_test_size - 1; begin += pfor_buffer_test_size / 10 + 1) {
        T step;
        for (step = 1; step < pfor_buffer_test_size; step++) {
            std::memset(pfor_buffer, 0, pfor_buffer_actual_size * sizeof(std::size_t));
            if (step == 1){
                invoke_for(begin, pfor_buffer_test_size, TestFunctor<T>(), p);
            } else {
                invoke_for(begin, pfor_buffer_test_size, step, TestFunctor<T>(), p);
            }
            // Verifying that parallel_for processed all items it should
            for (T i = begin; i < pfor_buffer_test_size; i = i + step) {
                if (pfor_buffer[i] != 1) {
                    CHECK_MESSAGE(false, "parallel_for didn't process all required elements");
                }
                pfor_buffer[i] = 0;
            }
            // Verifying that no extra items were processed and right bound of array wasn't crossed
            for (T i = 0; i < pfor_buffer_actual_size; i++) {
                if (pfor_buffer[i] != 0) {
                    CHECK_MESSAGE(false, "parallel_for processed an extra element");
                }
            }
        }
    }
}

template <typename Flavor, typename T>
void TestParallelForWithStepSupport() {
    static tbb::affinity_partitioner affinity_p;
    tbb::auto_partitioner auto_p;
    tbb::simple_partitioner simple_p;
    empty_partitioner_tag p;

    // Try out all partitioner combinations
    TestParallelForWithStepSupportHelper< Flavor,T,empty_partitioner_tag >(p);
    TestParallelForWithStepSupportHelper< Flavor,T,const tbb::auto_partitioner >(auto_p);
    TestParallelForWithStepSupportHelper< Flavor,T,const tbb::simple_partitioner >(simple_p);
    TestParallelForWithStepSupportHelper< Flavor,T,tbb::affinity_partitioner >(affinity_p);

    // Testing some corner cases
    tbb::parallel_for(static_cast<T>(2), static_cast<T>(1), static_cast<T>(1), TestFunctor<T>());
}

//! Test simple parallel_for with different partitioners
//! \brief \ref interface \ref requirement
TEST_CASE("Basic parallel_for") {
    std::atomic<unsigned long> counter{};
    const std::size_t number_of_partitioners = 5;
    const std::size_t iterations = 100000;

    tbb::parallel_for(std::size_t(0), iterations, [&](std::size_t) {
        counter++;
    });

    tbb::parallel_for(std::size_t(0), iterations, [&](std::size_t) {
        counter++;
    }, tbb::simple_partitioner());

    tbb::parallel_for(std::size_t(0), iterations, [&](std::size_t) {
        counter++;
    }, tbb::auto_partitioner());

    tbb::parallel_for(std::size_t(0), iterations, [&](std::size_t) {
        counter++;
    }, tbb::static_partitioner());

    tbb::affinity_partitioner aff;
    tbb::parallel_for(std::size_t(0), iterations, [&](std::size_t) {
        counter++;
    }, aff);

    CHECK_EQ(counter.load(std::memory_order_relaxed), iterations * number_of_partitioners);
}

//! Testing parallel for with different partitioners and ranges ranges
//! \brief \ref interface \ref requirement \ref stress
TEST_CASE("Flog test") {
    Flog<parallel_tag, 1>();
    Flog<parallel_tag, 10>();
    Flog<parallel_tag, 100>();
    Flog<parallel_tag, 1000>();
    Flog<parallel_tag, 10000>();
}

//! Testing parallel for with different types and step
//! \brief \ref interface \ref requirement
TEST_CASE_TEMPLATE("parallel_for with step support", T, short, unsigned short, int, unsigned int,
                                    long, unsigned long, long long, unsigned long long, std::size_t) {
    // Testing with different integer types
    TestParallelForWithStepSupport<parallel_tag, T>();
}

//! Testing with different types of ranges and partitioners
//! \brief \ref interface \ref requirement
TEST_CASE("Testing parallel_for with partitioners") {
    using namespace test_partitioner_utils::interaction_with_range_and_partitioner;

    test_partitioner_utils::SimpleBody b;
    tbb::affinity_partitioner ap;

    parallel_for(Range1(true, false), b, ap);
    parallel_for(Range6(false, true), b, ap);

    parallel_for(Range1(false, true), b, tbb::simple_partitioner());
    parallel_for(Range6(false, true), b, tbb::simple_partitioner());

    parallel_for(Range1(false, true), b, tbb::auto_partitioner());
    parallel_for(Range6(false, true), b, tbb::auto_partitioner());
}

