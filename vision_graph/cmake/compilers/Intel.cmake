# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if (MSVC)
    include(${CMAKE_CURRENT_LIST_DIR}/MSVC.cmake)
    set(TBB_WARNING_LEVEL ${TBB_WARNING_LEVEL} /W3)
elseif (APPLE)
    include(${CMAKE_CURRENT_LIST_DIR}/AppleClang.cmake)
    set(TBB_COMMON_COMPILE_FLAGS ${TBB_COMMON_COMPILE_FLAGS} -fstack-protector -Wformat -Wformat-security
                                 $<$<NOT:$<CONFIG:Debug>>:-fno-omit-frame-pointer -qno-opt-report-embed -D_FORTIFY_SOURCE=2>)
else()
    include(${CMAKE_CURRENT_LIST_DIR}/GNU.cmake)
    set(TBB_LIB_LINK_FLAGS ${TBB_LIB_LINK_FLAGS} -static-intel -Wl,-z,relro,-z,now,)
    set(TBB_COMMON_COMPILE_FLAGS ${TBB_COMMON_COMPILE_FLAGS} -fstack-protector -Wformat -Wformat-security
                                 $<$<NOT:$<CONFIG:Debug>>:-qno-opt-report-embed -D_FORTIFY_SOURCE=2>
                                 $<$<EQUAL:${TBB_ARCH},32>:-falign-stack=maintain-16-byte>)
endif()
