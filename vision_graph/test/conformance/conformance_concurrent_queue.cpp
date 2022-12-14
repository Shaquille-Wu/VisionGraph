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

#include <common/test.h>
#include <common/utils.h>
#include <common/utils_report.h>
#include <common/custom_allocators.h>
#include <common/container_move_support.h>

#include <tbb/concurrent_queue.h>
#include <tbb/cache_aligned_allocator.h>
#include <type_traits>
#include <atomic>

//! \file conformance_concurrent_queue.cpp
//! \brief Test for [containers.concurrent_queue containers.concurrent_bounded_queue] specification

template <typename T>
using test_allocator = StaticSharedCountingAllocator<tbb::cache_aligned_allocator<T>>;

static constexpr std::size_t MinThread = 1;
static constexpr std::size_t MaxThread = 4;

static constexpr std::size_t MAXTHREAD = 256;

static constexpr std::size_t M = 10000;
static std::atomic<long> PopKind[3];

static int Sum[MAXTHREAD];

template<typename CQ, typename ValueType, typename CounterType>
void push(CQ& q, ValueType v, CounterType i) {
    switch (i % 3) {
        case 0: q.push( v); break;
        case 1: q.push( std::move(v)); break;
        case 2: q.emplace( v); break;
        default: CHECK(false); break;
    }
}

template<typename T>
class ConcQWithCapacity : public tbb::concurrent_queue<T, test_allocator<T>> {
    using base_type = tbb::concurrent_queue<T, test_allocator<T>>;
public:
    ConcQWithCapacity() : my_capacity( std::size_t(-1) / (sizeof(void*) + sizeof(T)) ) {}
    std::size_t size() const {
        return this->unsafe_size();
    }

    std::size_t capacity() const {
        return my_capacity;
    }

    void set_capacity( const std::size_t n ) {
        my_capacity = n;
    }

    bool try_push( const T& source ) {
        base_type::push( source);
        return source.get_serial() < my_capacity;
    }

    bool try_pop( T& dest ) {
        base_type::try_pop( dest);
        return dest.get_serial() < my_capacity;
    }

private:
    std::size_t my_capacity;
};

template<typename CQ, typename T>
void TestEmptyQueue() {
    const CQ queue;
    CHECK(queue.size() == 0);
    CHECK(queue.capacity()> 0);
    CHECK(size_t(queue.capacity())>= std::size_t(-1)/(sizeof(void*)+sizeof(T)));
}

void TestEmptiness() {
    TestEmptyQueue<ConcQWithCapacity<char>, char>();
    TestEmptyQueue<ConcQWithCapacity<move_support_tests::Foo>, move_support_tests::Foo>();
    TestEmptyQueue<tbb::concurrent_bounded_queue<char, test_allocator<char>>, char>();
    TestEmptyQueue<tbb::concurrent_bounded_queue<move_support_tests::Foo,
           test_allocator<move_support_tests::Foo>>, move_support_tests::Foo>();
}

template<typename CQ, typename T>
void TestFullQueue() {
    using allocator_type = decltype(std::declval<CQ>().get_allocator());

    for (std::size_t n = 0; n < 100; ++n) {
        allocator_type::init_counters();
        {
            CQ queue;
            queue.set_capacity(n);
            for (std::size_t i = 0; i <= n; ++i) {
                T f;
                f.set_serial(i);
                bool result = queue.try_push( f);
                CHECK((result == (i < n)));
            }

            for (std::size_t i = 0; i <= n; ++i) {
                T f;
                bool result = queue.try_pop(f);
                CHECK((result == (i < n)));
                CHECK((result == 0 || f.get_serial() == i));
            }
        }
        CHECK(allocator_type::items_allocated == allocator_type::items_freed);
        CHECK(allocator_type::allocations == allocator_type::frees);
    }
}

void TestFullness() {
    TestFullQueue<ConcQWithCapacity<move_support_tests::Foo>, move_support_tests::Foo>();
    TestFullQueue<tbb::concurrent_bounded_queue<move_support_tests::Foo, test_allocator<move_support_tests::Foo>>, move_support_tests::Foo>();
}

template<typename CQ>
void TestClear() {
    using allocator_type = decltype(std::declval<CQ>().get_allocator());
    allocator_type::init_counters();
    const std::size_t n = 5;

    CQ queue;
    const std::size_t q_capacity = 10;
    queue.set_capacity(q_capacity);

    for (std::size_t i = 0; i < n; ++i) {
        move_support_tests::Foo f;
        f.set_serial(i);
        queue.push(f);
    }

    CHECK(queue.size() == n);

    queue.clear();
    CHECK(queue.size()==0);
    for (std::size_t i = 0; i < n; ++i) {
        move_support_tests::Foo f;
        f.set_serial(i);
        queue.push( f);
    }

    CHECK(queue.size() == n);
    queue.clear();
    CHECK(queue.size() == 0);

    for (std::size_t i = 0; i < n; ++i) {
        move_support_tests::Foo f;
        f.set_serial(i);
        queue.push(f);
    }

    CHECK(queue.size()==n);
}

void TestClearWorks() {
    TestClear<ConcQWithCapacity<move_support_tests::Foo>>();
    TestClear<tbb::concurrent_bounded_queue<move_support_tests::Foo, test_allocator<move_support_tests::Foo>>>();
}

template<typename Iterator1, typename Iterator2>
void TestIteratorAux( Iterator1 i, Iterator2 j, int size ) {
    Iterator1 old_i; // assigned at first iteration below
    for (std::size_t k = 0; k < (std::size_t)size; ++k) {
        CHECK(i != j);
        CHECK(!(i == j));
        // Test "->"
        CHECK((k+1 == i->get_serial()));
        if (k & 1) {
            // Test post-increment
            move_support_tests::Foo f = *old_i++;
            CHECK((k + 1 == f.get_serial()));
            // Test assignment
            i = old_i;
        } else {
            // Test pre-increment
            if (k < std::size_t(size - 1)) {
                move_support_tests::Foo f = *++i;
                CHECK((k + 2 == f.get_serial()));
            } else ++i;
            // Test assignment
            old_i = i;
        }
    }
    CHECK(!(i != j));
    CHECK(i == j);
}

template<typename Iterator1, typename Iterator2>
void TestIteratorAssignment( Iterator2 j ) {
    Iterator1 i(j);
    CHECK(i == j);
    CHECK(!(i != j));

    Iterator1 k;
    k = j;
    CHECK(k == j);
    CHECK(!(k != j));
}

template<typename Iterator, typename T>
void TestIteratorTraits() {
    static_assert( std::is_same<typename Iterator::iterator_category, std::forward_iterator_tag>::value, "wrong iterator category");

    T x;

    typename Iterator::reference xr = x;
    typename Iterator::pointer xp = &x;
    CHECK((&xr == xp));
}

// Test the iterators for concurrent_queue
template <typename CQ>
void TestIterator() {
    CQ queue;
    const CQ& const_queue = queue;
    for (int j=0; j < 500; ++j) {
        TestIteratorAux( queue.unsafe_begin()      , queue.unsafe_end()      , j);
        TestIteratorAux( queue.unsafe_cbegin()      , queue.unsafe_cend()      , j);
        TestIteratorAux( const_queue.unsafe_begin(), const_queue.unsafe_end(), j);
        TestIteratorAux( const_queue.unsafe_begin(), queue.unsafe_end()      , j);
        TestIteratorAux( queue.unsafe_begin()      , const_queue.unsafe_end(), j);
        move_support_tests::Foo f;
        f.set_serial(j+1);
        queue.push(f);
    }
    TestIteratorAssignment<typename CQ::const_iterator>( const_queue.unsafe_begin());
    TestIteratorAssignment<typename CQ::const_iterator>( queue.unsafe_begin());
    TestIteratorAssignment<typename CQ::iterator>( queue.unsafe_begin());
    TestIteratorTraits<typename CQ::const_iterator, const move_support_tests::Foo>();
    TestIteratorTraits<typename CQ::iterator, move_support_tests::Foo>();
}

void TestQueueIteratorWorks() {
    TestIterator<tbb::concurrent_queue<move_support_tests::Foo, test_allocator<move_support_tests::Foo>>>();
    TestIterator<tbb::concurrent_bounded_queue<move_support_tests::Foo, test_allocator<move_support_tests::Foo>>>();
}

// Define wrapper classes to test tbb::concurrent_queue<T>
template<typename T, typename A = tbb::cache_aligned_allocator<T>>
class ConcQWithSizeWrapper : public tbb::concurrent_queue<T, A> {
public:
    ConcQWithSizeWrapper() {}
    ConcQWithSizeWrapper( const ConcQWithSizeWrapper& q ) : tbb::concurrent_queue<T, A>(q) {}
    ConcQWithSizeWrapper( const ConcQWithSizeWrapper& q, const A& a ) : tbb::concurrent_queue<T, A>(q, a) {}
    ConcQWithSizeWrapper( const A& a ) : tbb::concurrent_queue<T, A>( a ) {}

    ConcQWithSizeWrapper( ConcQWithSizeWrapper&& q ) : tbb::concurrent_queue<T>(std::move(q)) {}
    ConcQWithSizeWrapper( ConcQWithSizeWrapper&& q, const A& a )
        : tbb::concurrent_queue<T, A>(std::move(q), a) { }

    template<typename InputIterator>
    ConcQWithSizeWrapper( InputIterator begin, InputIterator end, const A& a = A() )
        : tbb::concurrent_queue<T, A>(begin, end, a) {}
    typename tbb::concurrent_queue<T, A>::size_type size() const { return this->unsafe_size(); }
};

enum state_type {
    LIVE = 0x1234,
    DEAD = 0xDEAD
};

class Bar {
    state_type state;
public:
    static std::size_t construction_num, destruction_num;
    std::ptrdiff_t my_id;
    Bar() : state(LIVE), my_id(-1)
    {}

    Bar( std::size_t _i ) : state(LIVE), my_id(_i) { construction_num++; }

    Bar( const Bar& a_bar ) : state(LIVE) {
        CHECK(a_bar.state == LIVE);
        my_id = a_bar.my_id;
        construction_num++;
    }

    ~Bar() {
        CHECK(state == LIVE);
        state = DEAD;
        my_id = DEAD;
        destruction_num++;
    }

    void operator=( const Bar& a_bar ) {
        CHECK(a_bar.state == LIVE);
        CHECK(state == LIVE);
        my_id = a_bar.my_id;
    }
    friend bool operator==( const Bar& bar1, const Bar& bar2 ) ;
};

std::size_t Bar::construction_num = 0;
std::size_t Bar::destruction_num = 0;

bool operator==( const Bar& bar1, const Bar& bar2 ) {
    CHECK(bar1.state == LIVE);
    CHECK(bar2.state == LIVE);
    return bar1.my_id == bar2.my_id;
}

class BarIterator {
    Bar* bar_ptr;
    BarIterator(Bar* bp_) : bar_ptr(bp_) {}
public:
    Bar& operator*() const {
        return *bar_ptr;
    }
    BarIterator& operator++() {
        ++bar_ptr;
        return *this;
    }
    Bar* operator++(int) {
        Bar* result = &operator*();
        operator++();
        return result;
    }
    friend bool operator==(const BarIterator& bia, const BarIterator& bib) ;
    friend bool operator!=(const BarIterator& bia, const BarIterator& bib) ;
    template<typename CQ, typename T, typename TIter, typename CQ_EX, typename T_EX>
    friend void TestConstructors ();
} ;

bool operator==(const BarIterator& bia, const BarIterator& bib) {
    return bia.bar_ptr==bib.bar_ptr;
}

bool operator!=(const BarIterator& bia, const BarIterator& bib) {
    return bia.bar_ptr!=bib.bar_ptr;
}


class Bar_exception : public std::bad_alloc {
public:
    virtual const char *what() const noexcept override { return "making the entry invalid"; }
    virtual ~Bar_exception() noexcept {}
};

class BarEx {
    static int count;
public:
    state_type state;
    typedef enum {
        PREPARATION,
        COPY_CONSTRUCT
    } mode_type;
    static mode_type mode;
    std::ptrdiff_t my_id;
    std::ptrdiff_t my_tilda_id;

    static int button;

    BarEx() : state(LIVE), my_id(-1), my_tilda_id(-1)
    {}

    BarEx(std::size_t _i) : state(LIVE), my_id(_i), my_tilda_id(my_id^(-1))
    {}

    BarEx( const BarEx& a_bar ) : state(LIVE) {
        CHECK(a_bar.state==LIVE);
        my_id = a_bar.my_id;
        if (mode == PREPARATION)
            if (!(++count % 100)) {
                TBB_TEST_THROW(Bar_exception());
            }
        my_tilda_id = a_bar.my_tilda_id;
    }

    ~BarEx() {
        CHECK(state == LIVE);
        state = DEAD;
        my_id = DEAD;
    }
    static void set_mode( mode_type m ) { mode = m; }

    void operator=( const BarEx& a_bar ) {
        CHECK(a_bar.state == LIVE);
        CHECK(state == LIVE);
        my_id = a_bar.my_id;
        my_tilda_id = a_bar.my_tilda_id;
    }

    friend bool operator==(const BarEx& bar1, const BarEx& bar2 ) ;
};

int BarEx::count = 0;
BarEx::mode_type BarEx::mode = BarEx::PREPARATION;

bool operator==(const BarEx& bar1, const BarEx& bar2) {
    CHECK(bar1.state == LIVE);
    CHECK(bar2.state == LIVE);
    CHECK(((bar1.my_id ^ bar1.my_tilda_id) == -1));
    CHECK(((bar2.my_id ^ bar2.my_tilda_id) == -1));
    return bar1.my_id == bar2.my_id && bar1.my_tilda_id == bar2.my_tilda_id;
}

template<typename CQ, typename T, typename TIter, typename CQ_EX, typename T_EX>
void TestConstructors () {
    CQ src_queue;
    typename CQ::const_iterator dqb;
    typename CQ::const_iterator dqe;
    typename CQ::const_iterator iter;

    for (std::size_t size = 0; size < 1001; ++size) {
        for (std::size_t i = 0; i < size; ++i)
            src_queue.push(T(i + (i ^ size)));
        typename CQ::const_iterator sqb( src_queue.unsafe_begin());
        typename CQ::const_iterator sqe( src_queue.unsafe_end()  );

        CQ dst_queue(sqb, sqe);
        CQ copy_with_alloc(src_queue, typename CQ::allocator_type());

        REQUIRE_MESSAGE(src_queue.size() == dst_queue.size(), "different size");
        REQUIRE_MESSAGE(src_queue.size() == copy_with_alloc.size(), "different size");

        src_queue.clear();
    }

    T bar_array[1001];
    for (std::size_t size=0; size < 1001; ++size) {
        for (std::size_t i=0; i < size; ++i) {
            bar_array[i] = T(i+(i^size));
        }

        const TIter sab(bar_array + 0);
        const TIter sae(bar_array + size);

        CQ dst_queue2(sab, sae);

        CHECK(size == dst_queue2.size());
        CHECK(sab == TIter(bar_array+0));
        CHECK(sae == TIter(bar_array+size));

        dqb = dst_queue2.unsafe_begin();
        dqe = dst_queue2.unsafe_end();
        TIter v_iter(sab);
        for (; dqb != dqe; ++dqb, ++v_iter) {
            REQUIRE_MESSAGE((*dqb == *v_iter), "unexpected element");
        }

        REQUIRE_MESSAGE(v_iter==sae, "different size?");
    }

    src_queue.clear();

    CQ dst_queue3(src_queue);
    CHECK(src_queue.size() == dst_queue3.size());
    CHECK(0 == dst_queue3.size());

    int k = 0;
    for (std::size_t i = 0; i < 1001; ++i) {
        T tmp_bar;
        src_queue.push(T(++k));
        src_queue.push(T(++k));
        src_queue.try_pop(tmp_bar);

        CQ dst_queue4( src_queue);

        CHECK(src_queue.size() == dst_queue4.size());

        dqb = dst_queue4.unsafe_begin();
        dqe = dst_queue4.unsafe_end();
        iter = src_queue.unsafe_begin();

        for (; dqb != dqe; ++dqb, ++iter) {
            REQUIRE_MESSAGE((*dqb == *iter), "unexpected element");
        }

        REQUIRE_MESSAGE(iter == src_queue.unsafe_end(), "different size?");
    }

    CQ dst_queue5(src_queue);

    CHECK(src_queue.size() == dst_queue5.size());
    dqb = dst_queue5.unsafe_begin();
    dqe = dst_queue5.unsafe_end();
    iter = src_queue.unsafe_begin();
    for (; dqb != dqe; ++dqb, ++iter) {
        REQUIRE_MESSAGE(*dqb == *iter, "unexpected element");
    }

    for (std::size_t i=0; i<100; ++i) {
        T tmp_bar;
        src_queue.push(T(i + 1000));
        src_queue.push(T(i + 1000));
        src_queue.try_pop(tmp_bar);

        dst_queue5.push(T(i + 1000));
        dst_queue5.push(T(i + 1000));
        dst_queue5.try_pop(tmp_bar);
    }

    CHECK(src_queue.size() == dst_queue5.size());
    dqb = dst_queue5.unsafe_begin();
    dqe = dst_queue5.unsafe_end();
    iter = src_queue.unsafe_begin();
    for (; dqb != dqe; ++dqb, ++iter) {
        REQUIRE_MESSAGE((*dqb == *iter), "unexpected element");
    }

    REQUIRE_MESSAGE(iter == src_queue.unsafe_end(), "different size?");

#if TBB_USE_EXCEPTIONS
    k = 0;
    typename CQ_EX::size_type n_elements = 0;
    CQ_EX src_queue_ex;
    for (std::size_t size = 0; size < 1001; ++size) {
        T_EX tmp_bar_ex;
        typename CQ_EX::size_type n_successful_pushes = 0;
        T_EX::set_mode(T_EX::PREPARATION);
        try {
            src_queue_ex.push(T_EX(k + (k ^ size)));
            ++n_successful_pushes;
        } catch (...) {
        }
        ++k;
        try {
            src_queue_ex.push(T_EX(k + (k ^ size)));
            ++n_successful_pushes;
        } catch (...) {
        }
        ++k;
        src_queue_ex.try_pop(tmp_bar_ex);
        n_elements += (n_successful_pushes - 1);
        CHECK(src_queue_ex.size() == n_elements);

        T_EX::set_mode(T_EX::COPY_CONSTRUCT);
        CQ_EX dst_queue_ex(src_queue_ex);

        CHECK(src_queue_ex.size() == dst_queue_ex.size());

        typename CQ_EX::const_iterator dqb_ex = dst_queue_ex.unsafe_begin();
        typename CQ_EX::const_iterator dqe_ex = dst_queue_ex.unsafe_end();
        typename CQ_EX::const_iterator iter_ex = src_queue_ex.unsafe_begin();

        for (; dqb_ex != dqe_ex; ++dqb_ex, ++iter_ex) {
            REQUIRE_MESSAGE(*dqb_ex == *iter_ex, "unexpected element");
        }

        REQUIRE_MESSAGE(iter_ex==src_queue_ex.unsafe_end(), "different size?");
    }
#endif
    src_queue.clear();

    using qsize_t = typename CQ::size_type;
    for (qsize_t size = 0; size < 1001; ++size) {
        for (qsize_t i = 0; i < size; ++i) {
            src_queue.push(T(i + (i ^ size)));
        }
        std::vector<const T*> locations(size);
        typename CQ::const_iterator qit = src_queue.unsafe_begin();
        for (qsize_t i = 0; i < size; ++i, ++qit) {
            locations[i] = &(*qit);
        }

        qsize_t size_of_queue = src_queue.size();
        CQ dst_queue(std::move(src_queue));

        REQUIRE_MESSAGE((src_queue.empty() && src_queue.size() == 0), "not working move constructor?");
        REQUIRE_MESSAGE((size == size_of_queue && size_of_queue == qsize_t(dst_queue.size())), "not working move constructor?");

        qit = dst_queue.unsafe_begin();
        for (qsize_t i = 0; i < size; ++i, ++qit) {
            REQUIRE_MESSAGE(locations[i] == &(*qit), "there was data movement during move constructor");
        }

        for (qsize_t i = 0; i < size; ++i) {
            T test(i + (i ^ size));
            T popped;
            bool pop_result = dst_queue.try_pop( popped);

            CHECK(pop_result);
            CHECK(test == popped);
        }
    }
}

void TestQueueConstructors() {
    TestConstructors<ConcQWithSizeWrapper<Bar>, Bar, BarIterator, ConcQWithSizeWrapper<BarEx>, BarEx>();
    TestConstructors<tbb::concurrent_bounded_queue<Bar>, Bar, BarIterator, tbb::concurrent_bounded_queue<BarEx>, BarEx>();
}

template<typename T>
struct TestNegativeQueueBody {
    tbb::concurrent_bounded_queue<T>& queue;
    const std::size_t nthread;
    TestNegativeQueueBody( tbb::concurrent_bounded_queue<T>& q, std::size_t n ) : queue(q), nthread(n) {}
    void operator()( std::size_t k ) const {
        if (k == 0) {
            int number_of_pops = int(nthread) - 1;
            // Wait for all pops to pend.
            while (int(queue.size())> -number_of_pops) {
                std::this_thread::yield();
            }

            for (int i = 0; ; ++i) {
                CHECK(queue.size() == std::size_t(i - number_of_pops));
                CHECK((queue.empty() == (queue.size() <= 0)));
                if (i == number_of_pops) break;
                // Satisfy another pop
                queue.push(T());
            }
        } else {
            // Pop item from queue
            T item;
            queue.pop(item);
        }
    }
};

//! Test a queue with a negative size.
template<typename T>
void TestNegativeQueue( std::size_t nthread ) {
    tbb::concurrent_bounded_queue<T> queue;
    utils::NativeParallelFor( nthread, TestNegativeQueueBody<T>(queue,nthread));
}

template<typename T>
class ConcQPushPopWrapper : public tbb::concurrent_queue<T, test_allocator<T>> {
public:
    ConcQPushPopWrapper() : my_capacity(std::size_t(-1) / (sizeof(void*) + sizeof(T)))
    {}

    std::size_t size() const { return this->unsafe_size(); }
    void set_capacity( const ptrdiff_t n ) { my_capacity = n; }
    bool try_push( const T& source ) { return this->push( source); }
    bool try_pop( T& dest ) { return this->tbb::concurrent_queue<T, test_allocator<T>>::try_pop(dest); }
    std::size_t my_capacity;
};

template<typename CQ, typename T>
struct Body {
    CQ* queue;
    const std::size_t nthread;
    Body( std::size_t nthread_ ) : nthread(nthread_) {}
    void operator()( std::size_t thread_id ) const {
        long pop_kind[3] = {0, 0, 0};
        std::size_t serial[MAXTHREAD + 1];
        memset(serial, 0, nthread * sizeof(std::size_t));
        CHECK(thread_id < nthread);

        long sum = 0;
        for (std::size_t j = 0; j < M; ++j) {
            T f;
            f.set_thread_id(move_support_tests::serial_dead_state);
            f.set_serial(move_support_tests::serial_dead_state);
            bool prepopped = false;
            if (j & 1) {
                prepopped = queue->try_pop(f);
                ++pop_kind[prepopped];
            }
            T g;
            g.set_thread_id(thread_id);
            g.set_serial(j + 1);
            push(*queue, g, j);
            if (!prepopped) {
                while(!(queue)->try_pop(f)) std::this_thread::yield();
                ++pop_kind[2];
            }
            CHECK(f.get_thread_id() <= nthread);
            REQUIRE_MESSAGE((f.get_thread_id() == nthread || serial[f.get_thread_id()] < f.get_serial()), "partial order violation");
            serial[f.get_thread_id()] = f.get_serial();
            sum += int(f.get_serial() - 1);
        }
        Sum[thread_id] = sum;
        for (std::size_t k = 0; k < 3; ++k)
            PopKind[k] += pop_kind[k];
    }
};

template<typename CQ, typename T>
void TestPushPop( std::size_t prefill, std::ptrdiff_t capacity, std::size_t nthread ) {
    using allocator_type = decltype(std::declval<CQ>().get_allocator());
    CHECK(nthread> 0);
    std::ptrdiff_t signed_prefill = std::ptrdiff_t(prefill);

    if (signed_prefill + 1>= capacity) {
        return;
    }

    bool success = false;
    for (std::size_t k=0; k < 3; ++k) {
        PopKind[k] = 0;
    }

    for (std::size_t trial = 0; !success; ++trial) {
        allocator_type::init_counters();
        Body<CQ,T> body(nthread);
        CQ queue;
        queue.set_capacity(capacity);
        body.queue = &queue;
        for (std::size_t i = 0; i < prefill; ++i) {
            T f;
            f.set_thread_id(nthread);
            f.set_serial(1 + i);
            push(queue, f, i);
            CHECK(queue.size() == i + 1);
            CHECK(!queue.empty());
        }

        utils::NativeParallelFor( nthread, body);

        int sum = 0;
        for (std::size_t k = 0; k < nthread; ++k) {
            sum += Sum[k];
        }

        int expected = int( nthread * ((M - 1) * M / 2) + ((prefill - 1) * prefill) / 2);
        for (int i = int(prefill); --i>=0;) {
            CHECK(!queue.empty());
            T f;
            bool result = queue.try_pop(f);
            CHECK(result);
            CHECK(int(queue.size()) == i);
            sum += int(f.get_serial()) - 1;
        }
        REQUIRE_MESSAGE(queue.empty(), "The queue should be empty");
        REQUIRE_MESSAGE(queue.size() == 0, "The queue should have zero size");
        if (sum != expected) {
            REPORT("sum=%d expected=%d\n",sum,expected);
        }

        success = true;
        if (nthread> 1 && prefill == 0) {
            // Check that pop_if_present got sufficient exercise
            for (std::size_t k = 0; k < 2; ++k) {
                const int min_requirement = 100;
                const int max_trial = 20;

                if (PopKind[k] < min_requirement) {
                    if (trial>= max_trial) {
                        REPORT("Warning: %d threads had only %ld pop_if_present operations %s after %d trials (expected at least %d). "
                            "This problem may merely be unlucky scheduling. "
                            "Investigate only if it happens repeatedly.\n",
                            nthread, long(PopKind[k]), k==0?"failed":"succeeded", max_trial, min_requirement);
                    } else {
                        success = false;
                    }
               }
            }
        }
    }
}

void TestConcurrentPushPop() {
    for (std::size_t nthread = MinThread; nthread <= MaxThread; ++nthread) {
        INFO(" Testing with "<< nthread << " thread(s)");
        TestNegativeQueue<move_support_tests::Foo>(nthread);
        for (std::size_t prefill=0; prefill < 64; prefill += (1 + prefill / 3)) {
            TestPushPop<ConcQPushPopWrapper<move_support_tests::Foo>, move_support_tests::Foo>(prefill, std::ptrdiff_t(-1), nthread);
            TestPushPop<ConcQPushPopWrapper<move_support_tests::Foo>, move_support_tests::Foo>(prefill, std::ptrdiff_t(1), nthread);
            TestPushPop<ConcQPushPopWrapper<move_support_tests::Foo>, move_support_tests::Foo>(prefill, std::ptrdiff_t(2), nthread);
            TestPushPop<ConcQPushPopWrapper<move_support_tests::Foo>, move_support_tests::Foo>(prefill, std::ptrdiff_t(10), nthread);
            TestPushPop<ConcQPushPopWrapper<move_support_tests::Foo>, move_support_tests::Foo>(prefill, std::ptrdiff_t(100), nthread);
        }
        for (std::size_t prefill = 0; prefill < 64; prefill += (1 + prefill / 3) ) {
            TestPushPop<tbb::concurrent_bounded_queue<move_support_tests::Foo, test_allocator<move_support_tests::Foo>>,
                move_support_tests::Foo>(prefill, std::ptrdiff_t(-1), nthread);
            TestPushPop<tbb::concurrent_bounded_queue<move_support_tests::Foo, test_allocator<move_support_tests::Foo>>,
                move_support_tests::Foo>(prefill, std::ptrdiff_t(1), nthread);
            TestPushPop<tbb::concurrent_bounded_queue<move_support_tests::Foo, test_allocator<move_support_tests::Foo>>,
                move_support_tests::Foo>(prefill, std::ptrdiff_t(2), nthread);
            TestPushPop<tbb::concurrent_bounded_queue<move_support_tests::Foo, test_allocator<move_support_tests::Foo>>,
                move_support_tests::Foo>(prefill, std::ptrdiff_t(10), nthread);
            TestPushPop<tbb::concurrent_bounded_queue<move_support_tests::Foo, test_allocator<move_support_tests::Foo>>,
                move_support_tests::Foo>(prefill, std::ptrdiff_t(100), nthread);
        }
    }
}

class Foo_exception : public std::bad_alloc {
public:
    virtual const char *what() const throw() override { return "out of Foo limit"; }
    virtual ~Foo_exception() throw() {}
};

#if TBB_USE_EXCEPTIONS
static std::atomic<long> FooExConstructed;
static std::atomic<long> FooExDestroyed;
static std::atomic<long> serial_source;
static long MaxFooCount = 0;
static const long Threshold = 400;

class FooEx {
    state_type state;
public:
    int serial;
    FooEx() : state(LIVE) {
        ++FooExConstructed;
        serial = serial_source++;
    }

    FooEx( const FooEx& item ) : state(LIVE) {
        CHECK(item.state == LIVE);
        ++FooExConstructed;
        if (MaxFooCount && (FooExConstructed - FooExDestroyed) >= MaxFooCount) { // in push()
            throw Foo_exception();
        }
        serial = item.serial;
    }

    ~FooEx() {
        CHECK(state==LIVE);
        ++FooExDestroyed;
        state=DEAD;
        serial=DEAD;
    }

    void operator=( FooEx& item ) {
        CHECK(item.state==LIVE);
        CHECK(state==LIVE);
        serial = item.serial;
        if( MaxFooCount==2*Threshold && (FooExConstructed-FooExDestroyed) <= MaxFooCount/4 ) // in pop()
            throw Foo_exception();
    }

    void operator=( FooEx&& item ) {
        operator=( item );
        item.serial = 0;
    }

};

template <template <typename, typename> class CQ, typename A1, typename A2, typename T>
void TestExceptionBody() {
    enum methods {
        m_push = 0,
        m_pop
    };

    const int N = 1000;     // # of bytes

    MaxFooCount = 5;

    try {
        int n_pushed=0, n_popped=0;
        for(int t = 0; t <= 1; t++)// exception type -- 0 : from allocator(), 1 : from Foo's constructor
        {
            CQ<T,A1> queue_test;
            for( int m=m_push; m<=m_pop; m++ ) {
                // concurrent_queue internally rebinds the allocator to the one for 'char'
                A2::init_counters();

                if(t) MaxFooCount = MaxFooCount + 400;
                else A2::set_limits(N/2);

                try {
                    switch(m) {
                    case m_push:
                        for( int k=0; k<N; k++ ) {
                            push( queue_test, T(), k);
                            n_pushed++;
                        }
                        break;
                    case m_pop:
                        n_popped=0;
                        for( int k=0; k<n_pushed; k++ ) {
                            T elt;
                            queue_test.try_pop( elt);
                            n_popped++;
                        }
                        n_pushed = 0;
                        A2::set_limits();
                        break;
                    }
                    if( !t && m==m_push ) REQUIRE_MESSAGE(false, "should throw an exception");
                } catch ( Foo_exception & ) {
                    long tc = MaxFooCount;
                    MaxFooCount = 0; // disable exception
                    switch(m) {
                    case m_push:
                        REQUIRE_MESSAGE(ptrdiff_t(queue_test.size())==n_pushed, "incorrect queue size");
                        for( int k=0; k<(int)tc; k++ ) {
                            push( queue_test, T(), k);
                            n_pushed++;
                        }
                        break;
                    case m_pop:
                        n_pushed -= (n_popped+1); // including one that threw the exception
                        REQUIRE_MESSAGE(n_pushed>=0, "n_pushed cannot be less than 0");
                        for( int k=0; k<1000; k++ ) {
                            push( queue_test, T(), k);
                            n_pushed++;
                        }
                        REQUIRE_MESSAGE(!queue_test.empty(), "queue must not be empty");
                        REQUIRE_MESSAGE(ptrdiff_t(queue_test.size())==n_pushed, "queue size must be equal to n pushed");
                        for( int k=0; k<n_pushed; k++ ) {
                            T elt;
                            queue_test.try_pop( elt);
                        }
                        REQUIRE_MESSAGE(queue_test.empty(), "queue must be empty");
                        REQUIRE_MESSAGE(queue_test.size()==0, "queue must be empty");
                        break;
                    }
                    MaxFooCount = tc;
                } catch ( std::bad_alloc & ) {
                    A2::set_limits(); // disable exception from allocator
                    std::size_t size = queue_test.size();
                    switch(m) {
                        case m_push:
                            REQUIRE_MESSAGE(size>0, "incorrect queue size");
                            break;
                        case m_pop:
                            if( !t ) REQUIRE_MESSAGE(false, "should not throw an exception");
                            break;
                    }
                }
                INFO("for t= " << t << "and m= " << m << " exception test passed");
            }
        }
    } catch(...) {
        REQUIRE_MESSAGE(false, "unexpected exception");
    }
}

void TestExceptions() {
    using allocator_t = StaticSharedCountingAllocator<tbb::cache_aligned_allocator<std::size_t>>;
    using allocator_char_t = StaticSharedCountingAllocator<tbb::cache_aligned_allocator<char>>;
    TestExceptionBody<ConcQWithSizeWrapper, allocator_t, allocator_char_t, FooEx>();
    TestExceptionBody<tbb::concurrent_bounded_queue, allocator_t, allocator_char_t, FooEx>();

}

std::atomic<std::size_t> num_pushed;
std::atomic<std::size_t> num_popped;
std::atomic<std::size_t> failed_pushes;
std::atomic<std::size_t> failed_pops;

class SimplePushBody {
    tbb::concurrent_bounded_queue<int>* q;
    std::size_t max;
public:
    SimplePushBody(tbb::concurrent_bounded_queue<int>* _q, std::size_t hi_thr) : q(_q), max(hi_thr) {}

    void operator()(std::size_t thread_id) const {
        if (thread_id == max) {
            while ( q->size() < std::ptrdiff_t(max) ) {
                std::this_thread::yield();
            }
            q->abort();
            return;
        }
        try {
            q->push(42);
            ++num_pushed;
        } catch (...) {
            ++failed_pushes;
        }
    }
};

class SimplePopBody {
    tbb::concurrent_bounded_queue<int>* q;
    std::ptrdiff_t max;
    std::ptrdiff_t prefill;
public:
    SimplePopBody(tbb::concurrent_bounded_queue<int>* _q, std::size_t hi_thr, std::size_t nitems)
    : q(_q), max(hi_thr), prefill(nitems) {}

    void operator()(std::size_t thread_id) const {
        int e;
        if (thread_id == std::size_t(max)) {
            while (q->size()> prefill - max) {
                std::this_thread::yield();
            }

            q->abort();
            return;
        }
        try {
            q->pop(e);
            ++num_popped;
        } catch ( ... ) {
            ++failed_pops;
        }
    }
};

void TestAbort() {
    for (std::size_t nthreads = MinThread; nthreads <= MaxThread; ++nthreads) {
        tbb::concurrent_bounded_queue<int> iq1;
        iq1.set_capacity(0);
        for (std::size_t i = 0; i < 10; ++i) {
            num_pushed.store(0, std::memory_order_relaxed);
            num_popped.store(0, std::memory_order_relaxed);
            failed_pushes.store(0, std::memory_order_relaxed);
            failed_pops.store(0, std::memory_order_relaxed);
            SimplePushBody my_push_body1(&iq1, nthreads);
            utils::NativeParallelFor(nthreads + 1, my_push_body1);
            REQUIRE_MESSAGE(num_pushed == 0, "no elements should have been pushed to zero-sized queue");
            REQUIRE_MESSAGE(failed_pushes == nthreads, "All threads should have failed to push an element to zero-sized queue");
            // Do not test popping each time in order to test queue destruction with no previous pops
            if (nthreads < (MaxThread + MinThread) / 2) {
                int e;
                bool queue_empty = !iq1.try_pop(e);
                REQUIRE_MESSAGE(queue_empty, "no elements should have been popped from zero-sized queue");
            }
        }

        tbb::concurrent_bounded_queue<int> iq2;
        iq2.set_capacity(2);
        for (std::size_t i=0; i < 10; ++i) {
            num_pushed.store(0, std::memory_order_relaxed);
            num_popped.store(0, std::memory_order_relaxed);
            failed_pushes.store(0, std::memory_order_relaxed);
            failed_pops.store(0, std::memory_order_relaxed);
            SimplePushBody my_push_body2(&iq2, nthreads);
            utils::NativeParallelFor(nthreads + 1, my_push_body2);
            REQUIRE_MESSAGE(num_pushed <= 2, "at most 2 elements should have been pushed to queue of size 2");
            if (nthreads>= 2)
                REQUIRE_MESSAGE(failed_pushes == nthreads - 2, "nthreads-2 threads should have failed to push an element to queue of size 2");
            int e;
            while (iq2.try_pop(e)) ;
        }

        tbb::concurrent_bounded_queue<int> iq3;
        iq3.set_capacity(2);
        for (std::size_t i = 0; i < 10; ++i) {
            num_pushed.store(0, std::memory_order_relaxed);
            num_popped.store(0, std::memory_order_relaxed);
            failed_pushes.store(0, std::memory_order_relaxed);
            failed_pops.store(0, std::memory_order_relaxed);
            iq3.push(42);
            iq3.push(42);
            SimplePopBody my_pop_body(&iq3, nthreads, 2);
            utils::NativeParallelFor( nthreads+1, my_pop_body );
            REQUIRE_MESSAGE(num_popped <= 2, "at most 2 elements should have been popped from queue of size 2");
            if (nthreads>= 2)
                REQUIRE_MESSAGE(failed_pops == nthreads - 2, "nthreads-2 threads should have failed to pop an element from queue of size 2");
            else {
                int e;
                iq3.pop(e);
            }
        }

        tbb::concurrent_bounded_queue<int> iq4;
        std::size_t cap = nthreads / 2;
        if (!cap) cap = 1;
        iq4.set_capacity(cap);
        for (int i=0; i<10; ++i) {
            num_pushed.store(0, std::memory_order_relaxed);
            num_popped.store(0, std::memory_order_relaxed);
            failed_pushes.store(0, std::memory_order_relaxed);
            failed_pops.store(0, std::memory_order_relaxed);
            SimplePushBody my_push_body2(&iq4, nthreads);
            utils::NativeParallelFor(nthreads + 1, my_push_body2);
            REQUIRE_MESSAGE(num_pushed <= cap, "at most cap elements should have been pushed to queue of size cap");
            if (nthreads>= cap)
                REQUIRE_MESSAGE(failed_pushes == nthreads-cap, "nthreads-cap threads should have failed to push an element to queue of size cap");
            SimplePopBody my_pop_body(&iq4, nthreads, num_pushed);
            utils::NativeParallelFor( nthreads+1, my_pop_body );
            REQUIRE_MESSAGE((int)num_popped <= cap, "at most cap elements should have been popped from queue of size cap");
            if (nthreads>= cap)
                REQUIRE_MESSAGE(failed_pops == nthreads-cap, "nthreads-cap threads should have failed to pop an element from queue of size cap");
            else {
                int e;
                while (iq4.try_pop(e)) ;
            }
        }
    }
}
#endif

template <template <typename...> class ContainerType>
void test_member_types() {
    using container_type = ContainerType<int>;
    static_assert(std::is_same<typename container_type::allocator_type, tbb::cache_aligned_allocator<int>>::value,
                  "Incorrect default template allocator");

    static_assert(std::is_same<typename container_type::value_type, int>::value,
                  "Incorrect container value_type member type");

    static_assert(std::is_signed<typename container_type::difference_type>::value,
                  "Incorrect container difference_type member type");

    using value_type = typename container_type::value_type;
    static_assert(std::is_same<typename container_type::reference, value_type&>::value,
                  "Incorrect container reference member type");
    static_assert(std::is_same<typename container_type::const_reference, const value_type&>::value,
                  "Incorrect container const_reference member type");
    using allocator_type = typename container_type::allocator_type;
    static_assert(std::is_same<typename container_type::pointer, typename std::allocator_traits<allocator_type>::pointer>::value,
                  "Incorrect container pointer member type");
    static_assert(std::is_same<typename container_type::const_pointer, typename std::allocator_traits<allocator_type>::const_pointer>::value,
                  "Incorrect container const_pointer member type");

    static_assert(utils::is_forward_iterator<typename container_type::iterator>::value,
                  "Incorrect container iterator member type");
    static_assert(!std::is_const<typename container_type::iterator::value_type>::value,
                  "Incorrect container iterator member type");
    static_assert(utils::is_forward_iterator<typename container_type::const_iterator>::value,
                  "Incorrect container const_iterator member type");
    static_assert(std::is_const<typename container_type::const_iterator::value_type>::value,
                  "Incorrect container iterator member type");
}

enum push_t { push_op, try_push_op };

template<push_t push_op>
struct pusher {
    template<typename CQ, typename VType>
    static bool push( CQ& queue, VType&& val ) {
        queue.push( std::forward<VType>( val ) );
        return true;
    }
};

template<>
struct pusher< try_push_op> {
    template<typename CQ, typename VType>
    static bool push( CQ& queue, VType&& val ) {
        return queue.try_push( std::forward<VType>( val ) );
    }
};

enum pop_t { pop_op, try_pop_op };

template<pop_t pop_op>
struct popper {
    template<typename CQ, typename VType>
    static bool pop( CQ& queue, VType&& val ) {
        if( queue.empty() ) return false;
        queue.pop( std::forward<VType>( val ) );
        return true;
    }
};

template<>
struct popper<try_pop_op> {
    template<typename CQ, typename VType>
    static bool pop( CQ& queue, VType&& val ) {
        return queue.try_pop( std::forward<VType>( val ) );
    }
};

struct MoveOperationTracker {
    static std::size_t copy_constructor_called_times;
    static std::size_t move_constructor_called_times;
    static std::size_t copy_assignment_called_times;
    static std::size_t move_assignment_called_times;

    MoveOperationTracker() {}
    MoveOperationTracker(const MoveOperationTracker&) {
        ++copy_constructor_called_times;
    }
    MoveOperationTracker(MoveOperationTracker&&) {
        ++move_constructor_called_times;
    }
    MoveOperationTracker& operator=(MoveOperationTracker const&) {
        ++copy_assignment_called_times;
        return *this;
    }
    MoveOperationTracker& operator=(MoveOperationTracker&&) {
        ++move_assignment_called_times;
        return *this;
    }
};

size_t MoveOperationTracker::copy_constructor_called_times = 0;
size_t MoveOperationTracker::move_constructor_called_times = 0;
size_t MoveOperationTracker::copy_assignment_called_times = 0;
size_t MoveOperationTracker::move_assignment_called_times = 0;

template <class CQ, push_t push_op, pop_t pop_op>
void TestMoveSupport() {
    std::size_t &mcct = MoveOperationTracker::move_constructor_called_times;
    std::size_t &ccct = MoveOperationTracker::copy_constructor_called_times;
    std::size_t &cact = MoveOperationTracker::copy_assignment_called_times;
    std::size_t &mact = MoveOperationTracker::move_assignment_called_times;
    mcct = ccct = cact = mact = 0;

    CQ q;

    REQUIRE_MESSAGE(mcct == 0, "Value must be zero-initialized");
    REQUIRE_MESSAGE(ccct == 0, "Value must be zero-initialized");
    CHECK(pusher<push_op>::push( q, MoveOperationTracker() ));
    REQUIRE_MESSAGE(mcct == 1, "Not working push(T&&) or try_push(T&&)?");
    REQUIRE_MESSAGE(ccct == 0, "Copying of arg occurred during push(T&&) or try_push(T&&)");

    MoveOperationTracker ob;
    CHECK(pusher<push_op>::push( q, std::move(ob) ));
    REQUIRE_MESSAGE(mcct == 2, "Not working push(T&&) or try_push(T&&)?");
    REQUIRE_MESSAGE(ccct == 0, "Copying of arg occurred during push(T&&) or try_push(T&&)");

    REQUIRE_MESSAGE(cact == 0, "Copy assignment called during push(T&&) or try_push(T&&)");
    REQUIRE_MESSAGE(mact == 0, "Move assignment called during push(T&&) or try_push(T&&)");

    bool result = popper<pop_op>::pop( q, ob );
    CHECK(result);
    REQUIRE_MESSAGE(cact == 0, "Copy assignment called during try_pop(T&&)");
    REQUIRE_MESSAGE(mact == 1, "Move assignment was not called during try_pop(T&&)");
}

void TestMoveSupportInPushPop() {
    TestMoveSupport<tbb::concurrent_queue<MoveOperationTracker>, push_op, try_pop_op>();
    TestMoveSupport<tbb::concurrent_bounded_queue<MoveOperationTracker>, push_op, pop_op>();
    TestMoveSupport<tbb::concurrent_bounded_queue<MoveOperationTracker>, try_push_op, try_pop_op>();
}

template<class T>
class allocator: public tbb::cache_aligned_allocator<T> {
public:
    std::size_t m_unique_id;

    allocator() : m_unique_id( 0 ) {}

    allocator(size_t unique_id) { m_unique_id = unique_id; }

    template<typename U>
    allocator(const allocator<U>& a) noexcept { m_unique_id = a.m_unique_id; }

    template<typename U>
    struct rebind { typedef allocator<U> other; };

    friend bool operator==(const allocator& lhs, const allocator& rhs) {
        return lhs.m_unique_id == rhs.m_unique_id;
    }
};

template <typename Queue>
void AssertEquality(Queue &q, const std::vector<typename Queue::value_type> &vec) {
    CHECK(q.size() == typename Queue::size_type(vec.size()));
    CHECK(std::equal(q.unsafe_begin(), q.unsafe_end(), vec.begin()));
}

template <typename Queue>
void AssertEmptiness(Queue &q) {
    CHECK(q.empty());
    CHECK(!q.size());
    typename Queue::value_type elem;
    CHECK(!q.try_pop(elem));
}

template <push_t push_op, typename Queue>
void FillTest(Queue &q, const std::vector<typename Queue::value_type> &vec) {
    for (typename std::vector<typename Queue::value_type>::const_iterator it = vec.begin(); it != vec.end(); ++it)
        CHECK(pusher<push_op>::push(q, *it));
    AssertEquality(q, vec);
}

template <pop_t pop_op, typename Queue>
void EmptyTest(Queue &q, const std::vector<typename Queue::value_type> &vec) {
    typedef typename Queue::value_type value_type;

    value_type elem;
    typename std::vector<value_type>::const_iterator it = vec.begin();
    while (popper<pop_op>::pop(q, elem)) {
        CHECK(elem == *it);
        ++it;
    }
    CHECK(it == vec.end());
    AssertEmptiness(q);
}

template <typename T, typename A>
void bounded_queue_specific_test(tbb::concurrent_queue<T, A> &, const std::vector<T> &) { /* do nothing */ }

template <typename T, typename A>
void bounded_queue_specific_test(tbb::concurrent_bounded_queue<T, A> &q, const std::vector<T> &vec) {
    typedef typename tbb::concurrent_bounded_queue<T, A>::size_type size_type;

    FillTest<try_push_op>(q, vec);
    tbb::concurrent_bounded_queue<T, A> q2 = q;
    EmptyTest<pop_op>(q, vec);

    // capacity
    q2.set_capacity(size_type(vec.size()));
    CHECK(q2.capacity() == size_type(vec.size()));
    CHECK(q2.size() == size_type(vec.size()));
    CHECK(!q2.try_push(vec[0]));
    q.abort();
}

// Checks operability of the queue the data was moved from
template<typename T, typename CQ>
void TestQueueOperabilityAfterDataMove( CQ& queue ) {
    const std::size_t size = 10;
    std::vector<T> v(size);
    for( std::size_t i = 0; i < size; ++i ) v[i] = T( i * i + i );

    FillTest<push_op>(queue, v);
    EmptyTest<try_pop_op>(queue, v);
    bounded_queue_specific_test(queue, v);
}

template<class CQ, class T>
void TestMoveConstructors() {
    T::construction_num = T::destruction_num = 0;
    CQ src_queue( allocator<T>(0) );
    const std::size_t size = 10;
    for( std::size_t i = 0; i < size; ++i )
        src_queue.push( T(i + (i ^ size)) );
    CHECK(T::construction_num == 2 * size);
    CHECK(T::destruction_num == size);

    const T* locations[size];
    typename CQ::const_iterator qit = src_queue.unsafe_begin();
    for( std::size_t i = 0; i < size; ++i, ++qit )
        locations[i] = &(*qit);

    // Ensuring allocation operation takes place during move when allocators are different
    T::construction_num = T::destruction_num = 0;
    CQ dst_queue( std::move(src_queue), allocator<T>(1) );
    CHECK(T::construction_num == size);
    CHECK(T::destruction_num == size * 2); // One item is used by the queue destructor

    TestQueueOperabilityAfterDataMove<T>( src_queue );

    qit = dst_queue.unsafe_begin();
    for( std::size_t i = 0; i < size; ++i, ++qit ) {
        REQUIRE_MESSAGE(locations[i] != &(*qit), "an item should have been copied but was not" );
        locations[i] = &(*qit);
    }

    T::construction_num = T::destruction_num = 0;
    // Ensuring there is no allocation operation during move with equal allocators
    CQ dst_queue2( std::move(dst_queue), allocator<T>(1) );
    CHECK(T::construction_num == 0);
    CHECK(T::destruction_num == 0);

    TestQueueOperabilityAfterDataMove<T>( dst_queue );

    qit = dst_queue2.unsafe_begin();
    for( std::size_t i = 0; i < size; ++i, ++qit ) {
        REQUIRE_MESSAGE(locations[i] == &(*qit), "an item should have been moved but was not" );
    }

    for( std::size_t i = 0; i < size; ++i) {
        T test(i + (i ^ size));
        T popped;
        bool pop_result = dst_queue2.try_pop( popped );
        CHECK(pop_result);
        CHECK(test == popped);
    }
    CHECK(dst_queue2.empty());
    CHECK(dst_queue2.size() == 0);
}

void TestMoveConstruction() {
    TestMoveConstructors<ConcQWithSizeWrapper<Bar, allocator<Bar>>, Bar>();
    TestMoveConstructors<tbb::concurrent_bounded_queue<Bar, allocator<Bar>>, Bar>();
}

class NonTrivialConstructorType {
public:
    NonTrivialConstructorType( int a = 0 ) : m_a( a ), m_str( "" ) {}
    NonTrivialConstructorType( const std::string& str ) : m_a( 0 ), m_str( str ) {}
    NonTrivialConstructorType( int a, const std::string& str ) : m_a( a ), m_str( str ) {}
    int get_a() const { return m_a; }
    std::string get_str() const { return m_str; }
private:
    int m_a;
    std::string m_str;
};

enum emplace_t { emplace_op, try_emplace_op };

template<emplace_t emplace_op>
struct emplacer {
    template<typename CQ, typename... Args>
    static void emplace( CQ& queue, Args&&... val ) { queue.emplace( std::forward<Args>( val )... ); }
};

template<>
struct emplacer <try_emplace_op> {
    template<typename CQ, typename... Args>
    static void emplace( CQ& queue, Args&&... val ) {
        bool result = queue.try_emplace( std::forward<Args>( val )... );
        REQUIRE_MESSAGE(result, "try_emplace error\n");
    }
};

template<typename CQ, emplace_t emplace_op>
void TestEmplaceInQueue() {
    CQ cq;
    std::string test_str = "I'm being emplaced!";
    {
        emplacer<emplace_op>::emplace( cq, 5 );
        CHECK(cq.size() == 1);
        NonTrivialConstructorType popped( -1 );
        bool result = cq.try_pop( popped );
        CHECK(result);
        CHECK(popped.get_a() == 5);
        CHECK(popped.get_str() == std::string( "" ));
    }

    CHECK(cq.empty());

    {
        NonTrivialConstructorType popped( -1 );
        emplacer<emplace_op>::emplace( cq, std::string(test_str) );
        bool result = cq.try_pop( popped );
        CHECK(result);
        CHECK(popped.get_a() == 0);
        CHECK(popped.get_str() == test_str);
    }

    CHECK(cq.empty());

    {
        NonTrivialConstructorType popped( -1, "" );
        emplacer<emplace_op>::emplace( cq, 5, std::string(test_str) );
        bool result = cq.try_pop( popped );
        CHECK(result);
        CHECK(popped.get_a() == 5);
        CHECK(popped.get_str() == test_str);
    }
}
void TestEmplace() {
    TestEmplaceInQueue<ConcQWithSizeWrapper<NonTrivialConstructorType>, emplace_op>();
    TestEmplaceInQueue<tbb::concurrent_bounded_queue<NonTrivialConstructorType>, emplace_op>();
    TestEmplaceInQueue<tbb::concurrent_bounded_queue<NonTrivialConstructorType>, try_emplace_op>();
}

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
template <template <typename...> typename TQueue>
void TestDeductionGuides() {
    using ComplexType = const std::string*;
    std::vector<ComplexType> v;

    // check TQueue(InputIterator, InputIterator)
    TQueue q1(v.begin(), v.end());
    static_assert(std::is_same<decltype(q1), TQueue<ComplexType>>::value);

    // check TQueue(InputIterator, InputIterator, Allocator)
    TQueue q2(v.begin(), v.end(), std::allocator<ComplexType>());
    static_assert(std::is_same<decltype(q2), TQueue<ComplexType, std::allocator<ComplexType>>>::value);

    // check TQueue(TQueue &)
    TQueue q3(q1);
    static_assert(std::is_same<decltype(q3), decltype(q1)>::value);

    // check TQueue(TQueue &, Allocator)
    TQueue q4(q2, std::allocator<ComplexType>());
    static_assert(std::is_same<decltype(q4), decltype(q2)>::value);

    // check TQueue(TQueue &&)
    TQueue q5(std::move(q1));
    static_assert(std::is_same<decltype(q5), decltype(q1)>::value);

    // check TQueue(TQueue &&, Allocator)
    TQueue q6(std::move(q4), std::allocator<ComplexType>());
    static_assert(std::is_same<decltype(q6), decltype(q4)>::value);
}
#endif


//! Test constructors
//! \brief \ref interface \ref requirement
TEST_CASE("testing constructors") {
    TestQueueConstructors();
}

//! Test work with empty queue
//! \brief \ref interface \ref requirement
TEST_CASE("testing work with empty queue") {
    TestEmptiness();
}

//! Test set capacity operation
//! \brief \ref interface \ref requirement
TEST_CASE("testing set capacity operation") {
    TestFullness();
}

//! Test clean operation
//! \brief \ref interface \ref requirement
TEST_CASE("testing clean operation") {
    TestClearWorks();
}

//! Test move constructors
//! \brief \ref interface \ref requirement
TEST_CASE("testing move constructor") {
    TestMoveConstruction();
}

//! Test move support in push and pop
//! \brief \ref requirement
TEST_CASE("testing move support in push and pop") {
    TestMoveSupportInPushPop();
}

//! Test emplace operation
//! \brief \ref interface \ref requirement
TEST_CASE("testing emplace") {
    TestEmplace();
}

//! Test concurrent_queues member types
//! \brief \ref interface \ref requirement
TEST_CASE("testing concurrent_queues member types"){
    test_member_types<tbb::concurrent_queue>();
    test_member_types<tbb::concurrent_bounded_queue>();

    // Test size_type
    static_assert(std::is_unsigned<typename tbb::concurrent_queue<int>::size_type>::value,
                  "Incorrect tbb::concurrent_queue::size_type member type");
    static_assert(std::is_signed<typename tbb::concurrent_bounded_queue<int>::size_type>::value,
                  "Incorrect tbb::concurrent_bounded_queue::size_type member type");
}

//! Test iterators
//! \brief \ref interface \ref requirement
TEST_CASE("testing iterators") {
    TestQueueIteratorWorks();
}

//! Test concurrent oprations support
//! \brief \ref requirement
TEST_CASE("testing concurrent oprations support") {
    TestConcurrentPushPop();
}

#if TBB_USE_EXCEPTIONS
//! Test exception safety
//! \brief \ref requirement
TEST_CASE("testing exception safety") {
    TestExceptions();
}

//! Test abort operation
//! \brief \ref interface \ref requirement
TEST_CASE("testing abort operation") {
    TestAbort();
}
#endif

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT
//! Test deduction guides
//! \brief \ref interface
TEST_CASE("testing deduction guides") {
    TestDeductionGuides<tbb::concurrent_queue>();
    TestDeductionGuides<tbb::concurrent_bounded_queue>();
}
#endif
