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

#include "concurrent_monitor.h"

namespace tbb {
namespace detail {
namespace r1 {

void concurrent_monitor::thread_context::init() {
    new (sema.begin()) binary_semaphore;
    ready = true;
}

concurrent_monitor::~concurrent_monitor() {
    abort_all();
    __TBB_ASSERT( waitset_ec.empty(), "waitset not empty?" );
}

void concurrent_monitor::prepare_wait( thread_context& thr, uintptr_t ctx ) {
    if( !thr.ready )
        thr.init();
    // this is good place to pump previous skipped wakeup
    else if( thr.skipped_wakeup ) {
        thr.skipped_wakeup = false;
        thr.semaphore().P();
    }
    thr.context = ctx;
    thr.in_waitset = true;
    {
        tbb::spin_mutex::scoped_lock l( mutex_ec );
        thr.epoch = epoch.load(std::memory_order_relaxed);
        waitset_ec.add( (waitset_t::node_t*)&thr );
    }
    atomic_fence(std::memory_order_seq_cst);
}

void concurrent_monitor::cancel_wait( thread_context& thr ) {
    // possible skipped wakeup will be pumped in the following prepare_wait()
    thr.skipped_wakeup = true;
    // try to remove node from waitset
    bool th_in_waitset = thr.in_waitset;
    if( th_in_waitset ) {
        tbb::spin_mutex::scoped_lock l( mutex_ec );
        if (thr.in_waitset) {
            waitset_ec.remove( (waitset_t::node_t&)thr );
            // node is removed from waitset, so there will be no wakeup
            thr.in_waitset = false;
            thr.skipped_wakeup = false;
        }
    }
}

void concurrent_monitor::notify_one_relaxed() {
    if( waitset_ec.empty() )
        return;
    waitset_node_t* n;
    const waitset_node_t* end = waitset_ec.end();
    {
        tbb::spin_mutex::scoped_lock l( mutex_ec );
        epoch.store( epoch.load( std::memory_order_relaxed ) + 1, std::memory_order_relaxed );
        n = waitset_ec.front();
        if( n!=end ) {
            waitset_ec.remove( *n );
            to_thread_context(n)->in_waitset = false;
        }
    }
    if( n!=end )
        to_thread_context(n)->semaphore().V();
}

void concurrent_monitor::notify_all_relaxed() {
    if( waitset_ec.empty() )
        return;
    waitset_t temp;
    const waitset_node_t* end;
    {
        tbb::spin_mutex::scoped_lock l( mutex_ec );
        epoch.store( epoch.load( std::memory_order_relaxed ) + 1, std::memory_order_relaxed );
        waitset_ec.flush_to( temp );
        end = temp.end();
        for( waitset_node_t* n=temp.front(); n!=end; n=n->next )
            to_thread_context(n)->in_waitset = false;
    }
    waitset_node_t* nxt;
    for( waitset_node_t* n=temp.front(); n!=end; n=nxt ) {
        nxt = n->next;
        to_thread_context(n)->semaphore().V();
    }
#if TBB_USE_ASSERT
    temp.clear();
#endif
}

void concurrent_monitor::abort_all_relaxed() {
    if( waitset_ec.empty() )
        return;
    waitset_t temp;
    const waitset_node_t* end;
    {
        tbb::spin_mutex::scoped_lock l( mutex_ec );
        epoch.store( epoch.load( std::memory_order_relaxed ) + 1, std::memory_order_relaxed );
        waitset_ec.flush_to( temp );
        end = temp.end();
        for( waitset_node_t* n=temp.front(); n!=end; n=n->next )
            to_thread_context(n)->in_waitset = false;
    }
    waitset_node_t* nxt;
    for( waitset_node_t* n=temp.front(); n!=end; n=nxt ) {
        nxt = n->next;
        to_thread_context(n)->aborted = true;
        to_thread_context(n)->semaphore().V();
    }
#if TBB_USE_ASSERT
    temp.clear();
#endif
}

} // namespace r1
} // namespace detail
} // namespace tbb

