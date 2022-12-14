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

#include "tbb/detail/_exception.h"
#include "tbb/detail/_assert.h"
#include "tbb/detail/_template_helpers.h"

#include <cstring>
#include <cstdio>
#include <stdexcept> // std::runtime_error
#include <new>
#include <stdexcept>

#define __TBB_STD_RETHROW_EXCEPTION_POSSIBLY_BROKEN                             \
    (__GLIBCXX__ && __TBB_GLIBCXX_VERSION>=40700 && __TBB_GLIBCXX_VERSION<60000 && TBB_USE_EXCEPTIONS)

#if __TBB_STD_RETHROW_EXCEPTION_POSSIBLY_BROKEN
// GCC ABI declarations necessary for a workaround
#include <cxxabi.h>
#endif

namespace tbb {
namespace detail {
namespace r1 {

const char* bad_last_alloc::what() const noexcept(true) { return "bad allocation in previous or concurrent attempt"; }
const char* user_abort::what() const noexcept(true) { return "User-initiated abort has terminated this operation"; }
const char* missing_wait::what() const noexcept(true) { return "wait() was not called on the structured_task_group"; }

#if TBB_USE_EXCEPTIONS
    #define DO_THROW(exc, init_args) throw exc init_args;
#else /* !TBB_USE_EXCEPTIONS */
    #define PRINT_ERROR_AND_ABORT(exc_name, msg) \
        std::fprintf (stderr, "Exception %s with message %s would have been thrown, "  \
            "if exception handling had not been disabled. Aborting.\n", exc_name, msg); \
        std::fflush(stderr); \
        std::abort();
    #define DO_THROW(exc, init_args) PRINT_ERROR_AND_ABORT(#exc, #init_args)
#endif /* !TBB_USE_EXCEPTIONS */

void throw_exception ( exception_id eid ) {
    switch ( eid ) {
    case exception_id::bad_alloc: DO_THROW(std::bad_alloc, () );
    case exception_id::bad_last_alloc: DO_THROW(bad_last_alloc, ());
    case exception_id::user_abort: DO_THROW( user_abort, () );
    case exception_id::nonpositive_step: DO_THROW(std::invalid_argument, ("Step must be positive") );
    case exception_id::out_of_range: DO_THROW(std::out_of_range, ("Index out of requested size range"));
    case exception_id::reservation_length_error: DO_THROW(std::length_error, ("Attempt to exceed implementation defined length limits"));
    case exception_id::missing_wait: DO_THROW(missing_wait, ());
    case exception_id::invalid_load_factor: DO_THROW(std::out_of_range, ("Invalid hash load factor"));
    case exception_id::invalid_key: DO_THROW(std::out_of_range, ("invalid key"));
    case exception_id::bad_tagged_msg_cast: DO_THROW(std::runtime_error, ("Illegal tagged_msg cast"));
    default: __TBB_ASSERT ( false, "Unknown exception ID" );
    }
}

/* The "what" should be fairly short, not more than about 128 characters.
   Because we control all the call sites to handle_perror, it is pointless
   to bullet-proof it for very long strings.

   Design note: ADR put this routine off to the side in tbb_misc.cpp instead of
   Task.cpp because the throw generates a pathetic lot of code, and ADR wanted
   this large chunk of code to be placed on a cold page. */
void handle_perror( int error_code, const char* what ) {
    char buf[256];
#if defined(_MSC_VER) && _MSC_VER < 1500
 #define snprintf _snprintf
#endif
    int written = std::snprintf(buf, sizeof(buf), "%s: %s", what, std::strerror( error_code ));
    // On overflow, the returned value exceeds sizeof(buf) (for GLIBC) or is negative (for MSVC).
    __TBB_ASSERT_EX( written>0 && written<(int)sizeof(buf), "Error description is too long" );
    // Ensure that buffer ends in terminator.
    buf[sizeof(buf)-1] = 0;
#if TBB_USE_EXCEPTIONS
    throw std::runtime_error(buf);
#else
    PRINT_ERROR_AND_ABORT( "runtime_error", buf);
#endif /* !TBB_USE_EXCEPTIONS */
}

#if __TBB_STD_RETHROW_EXCEPTION_POSSIBLY_BROKEN
// Runtime detection and workaround for the GCC bug 62258.
// The problem is that std::rethrow_exception() does not increment a counter
// of active exceptions, causing std::uncaught_exception() to return a wrong value.
// The code is created after, and roughly reflects, the workaround
// at https://gcc.gnu.org/bugzilla/attachment.cgi?id=34683

void fix_broken_rethrow() {
    struct gcc_eh_data {
        void *       caughtExceptions;
        unsigned int uncaughtExceptions;
    };
    gcc_eh_data* eh_data = punned_cast<gcc_eh_data*>( abi::__cxa_get_globals() );
    ++eh_data->uncaughtExceptions;
}

bool gcc_rethrow_exception_broken() {
    bool is_broken;
    __TBB_ASSERT( !std::uncaught_exception(),
        "gcc_rethrow_exception_broken() must not be called when an exception is active" );
    try {
        // Throw, catch, and rethrow an exception
        try {
            throw __TBB_GLIBCXX_VERSION;
        } catch(...) {
            std::rethrow_exception( std::current_exception() );
        }
    } catch(...) {
        // Check the bug presence
        is_broken = std::uncaught_exception();
    }
    if( is_broken ) fix_broken_rethrow();
    __TBB_ASSERT( !std::uncaught_exception(), NULL );
    return is_broken;
}
#else
void fix_broken_rethrow() {}
bool gcc_rethrow_exception_broken() { return false; }
#endif /* __TBB_STD_RETHROW_EXCEPTION_POSSIBLY_BROKEN */

} // namespace r1
} // namespace detail
} // namespace tbb

