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

#include "common/test.h"
#include "common/utils.h"

#include "tbb/parallel_for.h"
#include <tbb/spin_mutex.h>
#include <tbb/spin_rw_mutex.h>
#include <tbb/queuing_mutex.h>
#include <tbb/queuing_rw_mutex.h>
#include <tbb/null_mutex.h>
#include <tbb/null_rw_mutex.h>

#include <type_traits>

//! \file conformance_mutex.cpp
//! \brief Test for [mutex.spin_mutex mutex.spin_rw_mutex mutex.queuing_mutex mutex.queuing_rw_mutex mutex.speculative_spin_mutex mutex.speculative_spin_rw_mutex mutex.null_mutex mutex.null_rw_mutex] specifications

template<typename M>
struct Counter {
    using mutex_type = M;
    M mutex;
    volatile long value;
};

//! Generic test of a TBB mutex
/** Does not test features specific to reader-writer locks. */
template<typename M>
void GeneralTest(const char* mutex_name, bool check = true) { // check flag is needed to disable correctness check for null mutexes (for test reusage)
    const int N = 100000;
    const int GRAIN = 10000;
    Counter<M> counter;
    counter.value = 0;

    // Stress test to force possible race condition of the counter
    utils::NativeParallelFor(N, GRAIN, [&] (int i) {
        if (i & 1) {
            // Try implicit acquire and explicit release
            typename M::scoped_lock lock(counter.mutex);
            counter.value = counter.value + 1;
            lock.release();
        } else {
            // Try explicit acquire and implicit release
            typename M::scoped_lock lock;
            lock.acquire(counter.mutex);
            counter.value = counter.value + 1;
        }
    });
    if (check) {
        REQUIRE_MESSAGE(counter.value == N, "ERROR for " << mutex_name << ": race is detected");
    }
}

//! Test try_acquire functionality of a non-reenterable mutex
template<typename M>
void TestTryAcquire(const char* mutex_name) {
    M tested_mutex;
    typename M::scoped_lock lock_outer;
    if (lock_outer.try_acquire(tested_mutex)) {
        lock_outer.release();
    } else {
        CHECK_MESSAGE(false, "ERROR for " << mutex_name << ": try_acquire failed though it should not");
    }
    {
        typename M::scoped_lock lock_inner(tested_mutex);
        CHECK_MESSAGE(!lock_outer.try_acquire(tested_mutex), "ERROR for " << mutex_name << ": try_acquire failed though it should not (1)");
    }
    if (lock_outer.try_acquire(tested_mutex)) {
        lock_outer.release();
    } else {
        CHECK_MESSAGE(false, "ERROR for " << mutex_name << ": try_acquire failed though it should not");
    }
}

//! Test try_acquire functionality of a non-reenterable mutex
template<typename M>
void TestTryAcquireReader(const char* mutex_name) {
    M tested_mutex;
    typename M::scoped_lock lock_outer;
    if (lock_outer.try_acquire(tested_mutex, false) ) {
        lock_outer.release();
    } else {
        CHECK_MESSAGE(false, "ERROR for " << mutex_name << ": try_acquire failed though it should not");
    }
    {
        typename M::scoped_lock lock_inner(tested_mutex, false); // read lock
        // try acquire on write
        CHECK_MESSAGE(!lock_outer.try_acquire(tested_mutex, true), "ERROR for " << mutex_name << ": try_acquire on write succeed though it should not (1)");
        lock_inner.release();                                    // unlock
        lock_inner.acquire(tested_mutex, true);                  // write lock
        // try acquire on read
        CHECK_MESSAGE(!lock_outer.try_acquire(tested_mutex, false), "ERROR for " << mutex_name << ": try_acquire on read succeed though it should not (2)");
    }
    if (lock_outer.try_acquire(tested_mutex, false) ) {
        lock_outer.release();
    } else {
        CHECK_MESSAGE(false, "ERROR for " << mutex_name << ": try_acquire failed though it should not");
    }
}

template<typename M, size_t N>
struct ArrayCounter {
    using mutex_type = M;
    M mutex;
    volatile long value[N];

    ArrayCounter() : value{0} {}

    void increment() {
        for (size_t k = 0; k < N; ++k) {
            ++value[k];
        }
    }

    bool value_is(long expected_value) const {
        for (size_t k = 0; k < N; ++k) {
            if (value[k] != expected_value) {
                return false;
            }
        }
        return true;
    }
};

template<typename M, typename Counter>
void TestReaderWriterLock_Impl(Counter& counter, typename M::scoped_lock& lock, const std::size_t i, const bool write) {
    bool okay = true;
    if (write) {
        long counter_value = counter.value[0];
        counter.increment();
        // Downgrade to reader
        if (i % 16 == 7) {
            if (!lock.downgrade_to_reader()) {
                // Get the previous value as downgrade with the same lock acquired was failed
                counter_value = counter.value[0] - 1;
            }
            okay = counter.value_is(counter_value + 1);
        }
    } else {
        okay = counter.value_is(counter.value[0]);
        // Upgrade to writer
        if (i % 8 == 3) {
            long counter_value = counter.value[0];
            if (!lock.upgrade_to_writer()) {
                // Failed to upgrade, reacquiring happened, need to update the value
                counter_value = counter.value[0];
            }
            counter.increment();
            okay = counter.value_is(counter_value + 1);
        }
    }
    CHECK_MESSAGE(okay, "Error in read write mutex operations");
}

//! Shared mutex type test
template<typename M>
void TestReaderWriterLock(const char* mutex_name) {
    ArrayCounter<M, 8> counter;
    const int N = 10000;
#if TBB_TEST_LOW_WORKLOAD
    const int GRAIN = 500;
#else
    const int GRAIN = 100;
#endif /* TBB_TEST_LOW_WORKLOAD */

    // Stress test similar to the general, but with upgrade/downgrade cases
    utils::NativeParallelFor(N, GRAIN, [&](int i) {
        //! Every 8th access is a write access
        const bool write = (i % 8) == 7;
        if (i & 1) {
            // Try implicit acquire and explicit release
            typename M::scoped_lock lock(counter.mutex, write);
            TestReaderWriterLock_Impl<M, ArrayCounter<M, 8>>(counter, lock, i, write);
            lock.release();
        } else {
            // Try explicit acquire and implicit release
            typename M::scoped_lock lock;
            lock.acquire(counter.mutex, write);
            TestReaderWriterLock_Impl<M, ArrayCounter<M, 8>>(counter, lock, i, write);
        }
    });
    // There is either a writer or a reader upgraded to a writer for each 4th iteration
    REQUIRE_MESSAGE(counter.value_is(N / 4), "ERROR for " << mutex_name << ": race is detected");
}

template<typename M>
void TestRWStateMultipleChange(const char* mutex_name) {
    static_assert(M::is_rw_mutex, "Incorrect mutex type");

    const int N = 1000;
    const int GRAIN = 100;
    M mutex;
    utils::NativeParallelFor(N, GRAIN, [&] (int) {
        typename M::scoped_lock l(mutex, /*write=*/false);
        for (int i = 0; i != GRAIN; ++i) {
            CHECK_MESSAGE(l.downgrade_to_reader(), mutex_name << " downgrade must succeed for read lock");
        }
        l.upgrade_to_writer();
        for (int i = 0; i != GRAIN; ++i) {
            CHECK_MESSAGE(l.upgrade_to_writer(), mutex_name << " upgrade must succeed for write lock");
        }
    });
}

//! Adaptor for using ISO C++0x style mutex as a TBB-style mutex.
template<typename M>
class TBB_MutexFromISO_Mutex {
    M my_iso_mutex;
public:
    typedef TBB_MutexFromISO_Mutex mutex_type;

    class scoped_lock;
    friend class scoped_lock;

    class scoped_lock {
        mutex_type* my_mutex;
        bool m_is_writer;
    public:
        scoped_lock() : my_mutex(NULL), m_is_writer(false) {}
        scoped_lock(mutex_type& m) : my_mutex(NULL), m_is_writer(false) {
            acquire(m);
        }
        scoped_lock(mutex_type& m, bool is_writer) : my_mutex(NULL) {
            acquire(m,is_writer);
        }
        void acquire(mutex_type& m) {
            m_is_writer = true;
            m.my_iso_mutex.lock();
            my_mutex = &m;
        }
        bool try_acquire(mutex_type& m) {
            m_is_writer = true;
            if (m.my_iso_mutex.try_lock()) {
                my_mutex = &m;
                return true;
            } else {
                return false;
            }
        }

        template<typename Q = M>
        typename std::enable_if<!Q::is_rw_mutex>::type release() {
            my_mutex->my_iso_mutex.unlock();
            my_mutex = NULL;
        }

        template<typename Q = M>
        typename std::enable_if<Q::is_rw_mutex>::type  release() {
            if (m_is_writer)
                my_mutex->my_iso_mutex.unlock();
            else
                my_mutex->my_iso_mutex.unlock_shared();
            my_mutex = NULL;
        }

        // Methods for reader-writer mutex
        // These methods can be instantiated only if M supports lock_shared() and try_lock_shared().

        void acquire(mutex_type& m, bool is_writer) {
            m_is_writer = is_writer;
            if (is_writer) m.my_iso_mutex.lock();
            else m.my_iso_mutex.lock_shared();
            my_mutex = &m;
        }
        bool try_acquire(mutex_type& m, bool is_writer) {
            m_is_writer = is_writer;
            if (is_writer ? m.my_iso_mutex.try_lock() : m.my_iso_mutex.try_lock_shared()) {
                my_mutex = &m;
                return true;
            } else {
                return false;
            }
        }
        bool upgrade_to_writer() {
            if (m_is_writer)
                return true;
            m_is_writer = true;
            my_mutex->my_iso_mutex.unlock_shared();
            my_mutex->my_iso_mutex.lock();
            return false;
        }
        bool downgrade_to_reader() {
            if (!m_is_writer)
                return true;
            m_is_writer = false;
            my_mutex->my_iso_mutex.unlock();
            my_mutex->my_iso_mutex.lock_shared();
            return false;
        }
        ~scoped_lock() {
            if (my_mutex)
                release();
        }
    };

    static constexpr bool is_recursive_mutex = M::is_recursive_mutex;
    static constexpr bool is_rw_mutex = M::is_rw_mutex;
};

template<typename C>
struct NullRecursive: utils::NoAssign {
    void recurse_till(std::size_t i, std::size_t till) const {
        if(i == till) {
            counter.value = counter.value + 1;
            return;
        }
        if(i & 1) {
            typename C::mutex_type::scoped_lock lock2(counter.mutex);
            recurse_till(i + 1, till);
            lock2.release();
        } else {
            typename C::mutex_type::scoped_lock lock2;
            lock2.acquire(counter.mutex);
            recurse_till(i + 1, till);
        }
    }

    void operator()(tbb::blocked_range<std::size_t>& range) const {
        typename C::mutex_type::scoped_lock lock(counter.mutex);
        recurse_till(range.begin(), range.end());
    }
    NullRecursive(C& counter_) : counter(counter_) {
        REQUIRE_MESSAGE(is_recursive_mutex, "Null mutex should be a recursive mutex.");
    }
    C& counter;
    bool is_recursive_mutex = C::mutex_type::is_recursive_mutex;
};

template<typename M>
struct NullUpgradeDowngrade: utils::NoAssign {
    void operator()(tbb::blocked_range<std::size_t>& range) const {
        typename M::scoped_lock lock2;
        for(std::size_t i = range.begin(); i != range.end(); ++i) {
            if(i & 1) {
                typename M::scoped_lock lock1(my_mutex, true);
                if(lock1.downgrade_to_reader() == false) {
                    REQUIRE_MESSAGE(false, "ERROR for " << mutex_name << ": downgrade should always succeed");
                }
            } else {
                lock2.acquire(my_mutex, false);
                if(lock2.upgrade_to_writer() == false) {
                    REQUIRE_MESSAGE(false, "ERROR for " << mutex_name << ": upgrade should always succeed");
                }
                lock2.release();
            }
        }
    }

    NullUpgradeDowngrade(M& m_, const char* n_) : my_mutex(m_), mutex_name(n_) {}
    M& my_mutex;
    const char* mutex_name;
};

template<typename M>
void TestNullMutex(const char* mutex_name) {
    INFO(mutex_name);
    Counter<M> counter;
    counter.value = 0;
    const std::size_t n = 100;
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, n, 10), NullRecursive<Counter<M>>(counter));
    M m;
    m.lock();
    REQUIRE(m.try_lock());
    m.unlock();
}

template<typename M>
void TestNullRWMutex(const char* mutex_name) {
    const std::size_t n = 100;
    M m;
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, n, 10), NullUpgradeDowngrade<M>(m, mutex_name));
    m.lock();
    REQUIRE(m.try_lock());
    m.lock_shared();
    REQUIRE(m.try_lock_shared());
    m.unlock_shared();
    m.unlock();
}

//! Testing Mutex requirements
//! \brief \ref interface \ref requirement
TEST_CASE("Basic Locable requirement test") {
    // BasicLockable
    GeneralTest<tbb::spin_mutex>("Spin Mutex");
    GeneralTest<tbb::spin_rw_mutex>("Spin RW Mutex");
    GeneralTest<tbb::queuing_mutex>("Queuing Mutex");
    GeneralTest<tbb::queuing_rw_mutex>("Queuing RW Mutex");
    GeneralTest<tbb::speculative_spin_mutex>("Speculative Spin Mutex");
    GeneralTest<tbb::speculative_spin_rw_mutex>("Speculative Spin RW Mutex");
    // NullMutexes
    GeneralTest<tbb::null_mutex>("Null Mutex", false);
    GeneralTest<tbb::null_rw_mutex>("Null RW Mutex", false);
    TestNullMutex<tbb::null_mutex>("Null Mutex");
    TestNullMutex<tbb::null_rw_mutex>("Null RW Mutex");
}

//! \brief \ref interface \ref requirement
TEST_CASE("Lockable requirement test") {
    // Lockable - single threaded try_acquire operations
    TestTryAcquire<tbb::spin_mutex>("Spin Mutex");
    TestTryAcquire<tbb::spin_rw_mutex>("Spin RW Mutex");
    TestTryAcquire<tbb::queuing_mutex>("Queuing Mutex");
    TestTryAcquire<tbb::queuing_rw_mutex>("Queuing RW Mutex");
    TestTryAcquire<tbb::speculative_spin_mutex>("Speculative Spin Mutex");
    TestTryAcquire<tbb::speculative_spin_rw_mutex>("Speculative Spin RW Mutex");
}

//! Testing ReaderWriterMutex requirements
//! \brief \ref interface \ref requirement
TEST_CASE("Shared mutexes (reader/writer) test") {
    // General reader writer capabilities + upgrade/downgrade
    TestReaderWriterLock<tbb::spin_rw_mutex>("Spin RW Mutex");
    TestReaderWriterLock<tbb::queuing_rw_mutex>("Queuing RW Mutex");
    TestReaderWriterLock<tbb::speculative_spin_rw_mutex>("Speculative Spin RW Mutex");
    TestNullRWMutex<tbb::null_rw_mutex>("Null RW Mutex");
    // Single threaded read/write try_acquire operations
    TestTryAcquireReader<tbb::spin_rw_mutex>("Spin RW Mutex");
    TestTryAcquireReader<tbb::queuing_rw_mutex>("Queuing RW Mutex");
    TestRWStateMultipleChange<tbb::spin_rw_mutex>("Spin RW Mutex");
    TestRWStateMultipleChange<tbb::queuing_rw_mutex>("Queuing RW Mutex");
    TestTryAcquireReader<tbb::speculative_spin_rw_mutex>("Speculative Spin RW Mutex");
    TestRWStateMultipleChange<tbb::speculative_spin_rw_mutex>("Speculative Spin RW Mutex");
}

//! Testing ISO C++ Mutex and Shared Mutex requirements.
//! Compatibility with the standard
//! \brief \ref interface \ref requirement
TEST_CASE("ISO interface test") {
    GeneralTest<TBB_MutexFromISO_Mutex<tbb::spin_mutex> >("ISO Spin Mutex");
    GeneralTest<TBB_MutexFromISO_Mutex<tbb::spin_rw_mutex> >("ISO Spin RW Mutex");
    TestTryAcquire<TBB_MutexFromISO_Mutex<tbb::spin_mutex> >("ISO Spin Mutex");
    TestTryAcquire<TBB_MutexFromISO_Mutex<tbb::spin_rw_mutex> >("ISO Spin RW Mutex");
    TestTryAcquireReader<TBB_MutexFromISO_Mutex<tbb::spin_rw_mutex> >("ISO Spin RW Mutex");
    TestReaderWriterLock<TBB_MutexFromISO_Mutex<tbb::spin_rw_mutex> >("ISO Spin RW Mutex");
}

