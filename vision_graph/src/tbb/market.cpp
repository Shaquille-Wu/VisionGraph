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

#include "tbb/global_control.h" // global_control::active_value

#include "market.h"
#include "main.h"
#include "governor.h"
#include "arena.h"
#include "thread_data.h"
#include "itt_notify.h"

#include <cstring> // std::memset()

namespace tbb {
namespace detail {
namespace r1 {

/** This method must be invoked under my_arenas_list_mutex. **/
arena* market::select_next_arena( arena* hint ) {
    unsigned next_arena_priority_level = num_priority_levels;
    if ( hint )
        next_arena_priority_level = hint->my_priority_level;
    for ( unsigned idx = 0; idx < next_arena_priority_level; ++idx ) {
        if ( !my_arenas[idx].empty() )
            return &*my_arenas[idx].begin();
    }
    // don't change if arena with higher priority is not found.
    return hint;
}

void market::insert_arena_into_list ( arena& a ) {
    __TBB_ASSERT( a.my_priority_level < num_priority_levels, nullptr );
    my_arenas[a.my_priority_level].push_front( a );
    __TBB_ASSERT( !my_next_arena || my_next_arena->my_priority_level < num_priority_levels, nullptr );
    my_next_arena = select_next_arena( my_next_arena );
}

void market::remove_arena_from_list ( arena& a ) {
    __TBB_ASSERT( a.my_priority_level < num_priority_levels, nullptr );
    my_arenas[a.my_priority_level].remove( a );
    if ( my_next_arena == &a )
        my_next_arena = nullptr;
    my_next_arena = select_next_arena( my_next_arena );
}

//------------------------------------------------------------------------
// market
//------------------------------------------------------------------------

market::market ( unsigned workers_soft_limit, unsigned workers_hard_limit, std::size_t stack_size )
    : my_num_workers_hard_limit(workers_hard_limit)
    , my_num_workers_soft_limit(workers_soft_limit)
    , my_next_arena(nullptr)
    , my_ref_count(1)
    , my_stack_size(stack_size)
    , my_workers_soft_limit_to_report(workers_soft_limit)
{
    // Once created RML server will start initializing workers that will need
    // global market instance to get worker stack size
    my_server = governor::create_rml_server( *this );
    __TBB_ASSERT( my_server, "Failed to create RML server" );
}

static unsigned calc_workers_soft_limit(unsigned workers_soft_limit, unsigned workers_hard_limit) {
    if( int soft_limit = market::app_parallelism_limit() )
        workers_soft_limit = soft_limit-1;
    else // if user set no limits (yet), use market's parameter
        workers_soft_limit = max( governor::default_num_threads() - 1, workers_soft_limit );
    if( workers_soft_limit >= workers_hard_limit )
        workers_soft_limit = workers_hard_limit-1;
    return workers_soft_limit;
}

market& market::global_market ( bool is_public, unsigned workers_requested, std::size_t stack_size ) {
    global_market_mutex_type::scoped_lock lock( theMarketMutex );
    market *m = theMarket;
    if( m ) {
        ++m->my_ref_count;
        const unsigned old_public_count = is_public? m->my_public_ref_count++ : /*any non-zero value*/1;
        lock.release();
        if( old_public_count==0 )
            set_active_num_workers( calc_workers_soft_limit(workers_requested, m->my_num_workers_hard_limit) );

        // do not warn if default number of workers is requested
        if( workers_requested != governor::default_num_threads()-1 ) {
            __TBB_ASSERT( skip_soft_limit_warning > workers_requested,
                          "skip_soft_limit_warning must be larger than any valid workers_requested" );
            unsigned soft_limit_to_report = m->my_workers_soft_limit_to_report.load(std::memory_order_relaxed);
            if( soft_limit_to_report < workers_requested ) {
                runtime_warning( "The number of workers is currently limited to %u. "
                                 "The request for %u workers is ignored. Further requests for more workers "
                                 "will be silently ignored until the limit changes.\n",
                                 soft_limit_to_report, workers_requested );
                // The race is possible when multiple threads report warnings.
                // We are OK with that, as there are just multiple warnings.
                unsigned expected_limit = soft_limit_to_report;
                m->my_workers_soft_limit_to_report.compare_exchange_strong(expected_limit, skip_soft_limit_warning);
            }

        }
        if( m->my_stack_size < stack_size )
            runtime_warning( "Thread stack size has been already set to %u. "
                             "The request for larger stack (%u) cannot be satisfied.\n",
                              m->my_stack_size, stack_size );
    }
    else {
        // TODO: A lot is done under theMarketMutex locked. Can anything be moved out?
        if( stack_size == 0 )
            stack_size = global_control::active_value(global_control::thread_stack_size);
        // Expecting that 4P is suitable for most applications.
        // Limit to 2P for large thread number.
        // TODO: ask RML for max concurrency and possibly correct hard_limit
        const unsigned factor = governor::default_num_threads()<=128? 4 : 2;
        // The requested number of threads is intentionally not considered in
        // computation of the hard limit, in order to separate responsibilities
        // and avoid complicated interactions between global_control and task_scheduler_init.
        // The market guarantees that at least 256 threads might be created.
        const unsigned workers_hard_limit = max(max(factor*governor::default_num_threads(), 256u), app_parallelism_limit());
        const unsigned workers_soft_limit = calc_workers_soft_limit(workers_requested, workers_hard_limit);
        // Create the global market instance
        std::size_t size = sizeof(market);
        __TBB_ASSERT( __TBB_offsetof(market, my_workers) + sizeof(thread_data*) == sizeof(market),
                      "my_workers must be the last data field of the market class");
        size += sizeof(thread_data*) * (workers_hard_limit - 1);
        __TBB_InitOnce::add_ref();
        void* storage = cache_aligned_allocate(size);
        std::memset( storage, 0, size );
        // Initialize and publish global market
        m = new (storage) market( workers_soft_limit, workers_hard_limit, stack_size );
        if( is_public )
            m->my_public_ref_count.store(1, std::memory_order_relaxed);
        theMarket = m;
        // This check relies on the fact that for shared RML default_concurrency==max_concurrency
        if ( !governor::UsePrivateRML && m->my_server->default_concurrency() < workers_soft_limit )
            runtime_warning( "RML might limit the number of workers to %u while %u is requested.\n"
                    , m->my_server->default_concurrency(), workers_soft_limit );
    }
    return *m;
}

void market::destroy () {
    this->market::~market(); // qualified to suppress warning
    cache_aligned_deallocate( this );
    __TBB_InitOnce::remove_ref();
}

bool market::release ( bool is_public, bool blocking_terminate ) {
    __TBB_ASSERT( theMarket == this, "Global market instance was destroyed prematurely?" );
    bool do_release = false;
    {
        global_market_mutex_type::scoped_lock lock( theMarketMutex );
        if ( blocking_terminate ) {
            __TBB_ASSERT( is_public, "Only an object with a public reference can request the blocking terminate" );
            while ( my_public_ref_count.load(std::memory_order_relaxed) == 1 &&
                    my_ref_count.load(std::memory_order_relaxed) > 1 ) {
                lock.release();
                // To guarantee that request_close_connection() is called by the last master, we need to wait till all
                // references are released. Re-read my_public_ref_count to limit waiting if new masters are created.
                // Theoretically, new private references to the market can be added during waiting making it potentially
                // endless.
                // TODO: revise why the weak scheduler needs market's pointer and try to remove this wait.
                // Note that the market should know about its schedulers for cancellation/exception/priority propagation,
                // see e.g. task_group_context::cancel_group_execution()
                while ( my_public_ref_count.load(std::memory_order_acquire) == 1 &&
                        my_ref_count.load(std::memory_order_acquire) > 1 ) {
                    yield();
                }
                lock.acquire( theMarketMutex );
            }
        }
        if ( is_public ) {
            __TBB_ASSERT( theMarket == this, "Global market instance was destroyed prematurely?" );
            __TBB_ASSERT( my_public_ref_count.load(std::memory_order_relaxed), NULL );
            --my_public_ref_count;
        }
        if ( --my_ref_count == 0 ) {
            __TBB_ASSERT( !my_public_ref_count.load(std::memory_order_relaxed), NULL );
            do_release = true;
            theMarket = NULL;
        }
    }
    if( do_release ) {
        __TBB_ASSERT( !my_public_ref_count.load(std::memory_order_relaxed),
            "No public references remain if we remove the market." );
        // inform RML that blocking termination is required
        my_join_workers = blocking_terminate;
        my_server->request_close_connection();
        return blocking_terminate;
    }
    return false;
}

int market::update_workers_request() {
    int old_request = my_num_workers_requested;
    my_num_workers_requested = min(my_total_demand,
                                   (int)my_num_workers_soft_limit.load(std::memory_order_relaxed));
#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
    if (my_mandatory_num_requested > 0) {
        __TBB_ASSERT(my_num_workers_soft_limit.load(std::memory_order_relaxed) == 0, NULL);
        my_num_workers_requested = 1;
    }
#endif
    update_allotment();
    return my_num_workers_requested - old_request;
}

void market::set_active_num_workers ( unsigned soft_limit ) {
    market *m;

    {
        global_market_mutex_type::scoped_lock lock( theMarketMutex );
        if ( !theMarket )
            return; // actual value will be used at market creation
        m = theMarket;
        if (m->my_num_workers_soft_limit.load(std::memory_order_relaxed) == soft_limit)
            return;
        ++m->my_ref_count;
    }
    // have my_ref_count for market, use it safely

    int delta = 0;
    {
        arenas_list_mutex_type::scoped_lock lock( m->my_arenas_list_mutex );
        __TBB_ASSERT(soft_limit <= m->my_num_workers_hard_limit, NULL);

#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
        arena_list_type* arenas = m->my_arenas;

        if (m->my_num_workers_soft_limit.load(std::memory_order_relaxed) == 0 &&
            m->my_mandatory_num_requested > 0)
        {
            for (unsigned level = 0; level < num_priority_levels; ++level )
                for (arena_list_type::iterator it = arenas[level].begin(); it != arenas[level].end(); ++it)
                    if (it->my_global_concurrency_mode.load(std::memory_order_relaxed))
                        m->disable_mandatory_concurrency_impl(&*it);
        }
        __TBB_ASSERT(m->my_mandatory_num_requested == 0, NULL);
#endif

        m->my_num_workers_soft_limit.store(soft_limit, std::memory_order_release);
        // report only once after new soft limit value is set
        m->my_workers_soft_limit_to_report.store(soft_limit, std::memory_order_relaxed);

#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
        if (m->my_num_workers_soft_limit.load(std::memory_order_relaxed) == 0) {
            for (unsigned level = 0; level < num_priority_levels; ++level )
                for (arena_list_type::iterator it = arenas[level].begin(); it != arenas[level].end(); ++it)
                    if (it->has_enqueued_tasks())
                        m->enable_mandatory_concurrency_impl(&*it);
        }
#endif

        delta = m->update_workers_request();
    }
    // adjust_job_count_estimate must be called outside of any locks
    if( delta!=0 )
        m->my_server->adjust_job_count_estimate( delta );
    // release internal market reference to match ++m->my_ref_count above
    m->release( /*is_public=*/false, /*blocking_terminate=*/false );
}

bool governor::does_client_join_workers (const rml::tbb_client &client) {
    return ((const market&)client).must_join_workers();
}

arena* market::create_arena ( int num_slots, int num_reserved_slots, unsigned arena_priority_level,
                              std::size_t stack_size )
{
    __TBB_ASSERT( num_slots > 0, NULL );
    __TBB_ASSERT( num_reserved_slots <= num_slots, NULL );
    // Add public market reference for master thread/task_arena (that adds an internal reference in exchange).
    market &m = global_market( /*is_public=*/true, num_slots-num_reserved_slots, stack_size );
    arena& a = arena::allocate_arena( m, num_slots, num_reserved_slots, arena_priority_level );
    // Add newly created arena into the existing market's list.
    arenas_list_mutex_type::scoped_lock lock(m.my_arenas_list_mutex);
    m.insert_arena_into_list(a);
    return &a;
}

/** This method must be invoked under my_arenas_list_mutex. **/
void market::detach_arena ( arena& a ) {
    __TBB_ASSERT( theMarket == this, "Global market instance was destroyed prematurely?" );
    __TBB_ASSERT( !a.my_slots[0].is_occupied(), NULL );
    if (a.my_global_concurrency_mode.load(std::memory_order_relaxed))
        disable_mandatory_concurrency_impl(&a);

    remove_arena_from_list(a);
    if ( a.my_aba_epoch == my_arenas_aba_epoch )
        ++my_arenas_aba_epoch;
}

void market::try_destroy_arena ( arena* a, uintptr_t aba_epoch ) {
    bool locked = true;
    __TBB_ASSERT( a, NULL );
    // we hold reference to the market, so it cannot be destroyed at any moment here
    __TBB_ASSERT( this == theMarket, NULL );
    __TBB_ASSERT( my_ref_count!=0, NULL );
    my_arenas_list_mutex.lock();
    assert_market_valid();
        arena_list_type::iterator it = my_arenas[a->my_priority_level].begin();
        for ( ; it != my_arenas[a->my_priority_level].end(); ++it ) {
            if ( a == &*it ) {
                if ( it->my_aba_epoch == aba_epoch ) {
                    // Arena is alive
                    if ( !a->my_num_workers_requested && !a->my_references.load(std::memory_order_relaxed) ) {
                        __TBB_ASSERT(
                            !a->my_num_workers_allotted &&
                            (a->my_pool_state == arena::SNAPSHOT_EMPTY || !a->my_max_num_workers),
                            "Inconsistent arena state"
                        );
                        // Arena is abandoned. Destroy it.
                        detach_arena( *a );
                        my_arenas_list_mutex.unlock();
                        locked = false;
                        a->free_arena();
                    }
                }
                if (locked)
                    my_arenas_list_mutex.unlock();
                return;
            }
        }
    my_arenas_list_mutex.unlock();
}

/** This method must be invoked under my_arenas_list_mutex. **/
arena* market::arena_in_need ( arena_list_type* arenas, arena* hint ) {
    // TODO: make sure arena with higher priority returned only if there are available slots in it.
    hint = select_next_arena( hint );
    if ( !hint )
        return nullptr;
    arena_list_type::iterator it = hint;
    unsigned curr_priority_level = hint->my_priority_level;
    __TBB_ASSERT( it != arenas[curr_priority_level].end(), nullptr );
    do {
        arena& a = *it;
        if ( ++it == arenas[curr_priority_level].end() ) {
            do {
                ++curr_priority_level %= num_priority_levels;
            } while ( arenas[curr_priority_level].empty() );
            it = arenas[curr_priority_level].begin();
        }
        if( a.num_workers_active() < a.my_num_workers_allotted ) {
            a.my_references += arena::ref_worker;
            return &a;
        }
    } while ( it != hint );
    return nullptr;
}

arena* market::arena_in_need(arena* prev) {
    atomic_fence(std::memory_order_acquire);
    if (my_total_demand <= 0)
        return nullptr;
    arenas_list_mutex_type::scoped_lock lock(my_arenas_list_mutex, /*is_writer=*/false);
    // TODO: introduce three state response: alive, not_alive, no_market_arenas
    if ( is_arena_alive(prev) )
        return arena_in_need(my_arenas, prev);
    return arena_in_need(my_arenas, my_next_arena);
}

int market::update_allotment ( arena_list_type* arenas, int workers_demand, int max_workers ) {
    __TBB_ASSERT( workers_demand > 0, nullptr );
    max_workers = min(workers_demand, max_workers);
    int unassigned_workers = max_workers;
    int assigned = 0;
    int carry = 0;
    for (unsigned list_idx = 0; list_idx < num_priority_levels; ++list_idx ) {
        int assigned_per_priority = 0;
        for (arena_list_type::iterator it = arenas[list_idx].begin(); it != arenas[list_idx].end(); ++it) {
            arena& a = *it;
            if (a.my_num_workers_requested <= 0) {
                __TBB_ASSERT(!a.my_num_workers_allotted, nullptr);
                continue;
            }
            int allotted = 0;
#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
            if (my_num_workers_soft_limit.load(std::memory_order_relaxed) == 0) {
                __TBB_ASSERT(max_workers == 0 || max_workers == 1, nullptr);
                allotted = a.my_global_concurrency_mode.load(std::memory_order_relaxed) &&
                    assigned < max_workers ? 1 : 0;
            } else
#endif
            {
                int tmp = a.my_num_workers_requested * unassigned_workers + carry;
                allotted = tmp / my_priority_level_demand[list_idx];
                carry = tmp % my_priority_level_demand[list_idx];
                // a.my_num_workers_requested may temporarily exceed a.my_max_num_workers
                allotted = min(allotted, (int)a.my_max_num_workers);
            }
            a.my_num_workers_allotted = allotted;
            assigned += allotted;
            assigned_per_priority += allotted;
        }
        unassigned_workers -= assigned_per_priority;
    }
    __TBB_ASSERT( 0 <= assigned && assigned <= max_workers, nullptr );
    return assigned;
}

/** This method must be invoked under my_arenas_list_mutex. **/
bool market::is_arena_in_list( arena_list_type &arenas, arena *a ) {
    __TBB_ASSERT( a, "Expected non-null pointer to arena." );
    for ( arena_list_type::iterator it = arenas.begin(); it != arenas.end(); ++it )
        if ( a == &*it )
            return true;
    return false;
}

/** This method must be invoked under my_arenas_list_mutex. **/
bool market::is_arena_alive(arena* a) {
    if ( !a )
        return false;

    // Still cannot access internals of the arena since the object itself might be destroyed.

    for ( unsigned idx = 0; idx < num_priority_levels; ++idx ) {
        if ( is_arena_in_list( my_arenas[idx], a ) )
            return true;
    }
    return false;
}

#if __TBB_ENQUEUE_ENFORCED_CONCURRENCY
void market::enable_mandatory_concurrency_impl ( arena *a ) {
    __TBB_ASSERT(!a->my_global_concurrency_mode.load(std::memory_order_relaxed), NULL);
    __TBB_ASSERT(my_num_workers_soft_limit.load(std::memory_order_relaxed) == 0, NULL);

    a->my_global_concurrency_mode.store(true, std::memory_order_relaxed);
    my_mandatory_num_requested++;
}

void market::enable_mandatory_concurrency ( arena *a ) {
    int delta = 0;
    {
        arenas_list_mutex_type::scoped_lock lock(my_arenas_list_mutex);
        if (my_num_workers_soft_limit.load(std::memory_order_relaxed) != 0 ||
            a->my_global_concurrency_mode.load(std::memory_order_relaxed))
            return;

        enable_mandatory_concurrency_impl(a);
        delta = update_workers_request();
    }

    if (delta != 0)
        my_server->adjust_job_count_estimate(delta);
}

void market::disable_mandatory_concurrency_impl(arena* a) {
    __TBB_ASSERT(a->my_global_concurrency_mode.load(std::memory_order_relaxed), NULL);
    __TBB_ASSERT(my_mandatory_num_requested > 0, NULL);

    a->my_global_concurrency_mode.store(false, std::memory_order_relaxed);
    my_mandatory_num_requested--;
}

void market::mandatory_concurrency_disable ( arena *a ) {
    int delta = 0;
    {
        arenas_list_mutex_type::scoped_lock lock(my_arenas_list_mutex);
        if (!a->my_global_concurrency_mode.load(std::memory_order_relaxed))
            return;
        // There is a racy window in advertise_new_work between mandtory concurrency enabling and 
        // setting SNAPSHOT_FULL. It gives a chance to spawn request to disable mandatory concurrency.
        // Therefore, we double check that there is no enqueued tasks.
        if (a->has_enqueued_tasks())
            return;

        __TBB_ASSERT(my_num_workers_soft_limit.load(std::memory_order_relaxed) == 0, NULL);
        disable_mandatory_concurrency_impl(a);

        delta = update_workers_request();
    }
    if (delta != 0)
        my_server->adjust_job_count_estimate(delta);
}
#endif /* __TBB_ENQUEUE_ENFORCED_CONCURRENCY */

void market::adjust_demand ( arena& a, int delta ) {
    __TBB_ASSERT( theMarket, "market instance was destroyed prematurely?" );
    if ( !delta )
        return;
    my_arenas_list_mutex.lock();
    int prev_req = a.my_num_workers_requested;
    a.my_num_workers_requested += delta;
    if ( a.my_num_workers_requested <= 0 ) {
        a.my_num_workers_allotted = 0;
        if ( prev_req <= 0 ) {
            my_arenas_list_mutex.unlock();
            return;
        }
        delta = -prev_req;
    }
    else if ( prev_req < 0 ) {
        delta = a.my_num_workers_requested;
    }
    my_total_demand += delta;
    my_priority_level_demand[a.my_priority_level] += delta;
    unsigned effective_soft_limit = my_num_workers_soft_limit.load(std::memory_order_relaxed);
    update_allotment();
    if ( delta > 0 ) {
        // can't overflow soft_limit, but remember values request by arenas in
        // my_total_demand to not prematurely release workers to RML
        if ( my_num_workers_requested+delta > (int)effective_soft_limit)
            delta = effective_soft_limit - my_num_workers_requested;
    } else {
        // the number of workers should not be decreased below my_total_demand
        if ( my_num_workers_requested+delta < my_total_demand )
            delta = min(my_total_demand, (int)effective_soft_limit) - my_num_workers_requested;
    }
    my_num_workers_requested += delta;
    __TBB_ASSERT( my_num_workers_requested <= (int)effective_soft_limit, NULL );

    my_arenas_list_mutex.unlock();
    // Must be called outside of any locks
    my_server->adjust_job_count_estimate( delta );
}

void market::process( job& j ) {
    thread_data& td = static_cast<thread_data&>(j);
    // td.my_arena can be dead. Don't access it until arena_in_need is called
    arena *a = td.my_arena;
    for (int i = 0; i < 2; ++i) {
        while ( (a = arena_in_need(a)) ) {
            a->process(td);
            a = nullptr; // to avoid double checks in arena_in_need(arena*) for the same priority level
        }
        // Workers leave market because there is no arena in need. It can happen earlier than
        // adjust_job_count_estimate() decreases my_slack and RML can put this thread to sleep.
        // It might result in a busy-loop checking for my_slack<0 and calling this method instantly.
        // the yield refines this spinning.
        if ( !i ) {
            yield();
        }
    }
}

void market::cleanup( job& j) {
    __TBB_ASSERT( theMarket != this, NULL );
    governor::auto_terminate(&j);
}

void market::acknowledge_close_connection() {
    destroy();
}

::rml::job* market::create_one_job() {
    unsigned short index = ++my_first_unused_worker_idx;
    __TBB_ASSERT( index > 0, NULL );
    ITT_THREAD_SET_NAME(_T("TBB Worker Thread"));
    // index serves as a hint decreasing conflicts between workers when they migrate between arenas
    thread_data* td = new(cache_aligned_allocate(sizeof(thread_data))) thread_data{ index, true };
    __TBB_ASSERT( index <= my_num_workers_hard_limit, NULL );
    __TBB_ASSERT( my_workers[index - 1] == nullptr, NULL );
    my_workers[index - 1] = td;
    return td;
}

void market::add_external_thread(thread_data& td) {
    context_state_propagation_mutex_type::scoped_lock lock(the_context_state_propagation_mutex);
    my_masters.push_front(td);
}

void market::remove_external_thread(thread_data& td) {
    context_state_propagation_mutex_type::scoped_lock lock(the_context_state_propagation_mutex);
    my_masters.remove(td);
}

} // namespace r1
} // namespace detail
} // namespace tbb
