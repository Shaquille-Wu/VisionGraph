cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)
set(CMAKE_VERBOSE_MAKEFILE  ON)

enable_testing()

add_library(gtest STATIC EXCLUDE_FROM_ALL gtest/gtest.h gtest/gtest-all.cpp)
target_include_directories(gtest PUBLIC "${CMAKE_SOURCE_DIR}/tests")
if (NOT ANDROID)
    target_link_libraries(gtest PUBLIC pthread)
else()
    add_definitions(-DGTEST_USE_OWN_TR1_TUPLE=1)
endif()

add_library(gtest_main EXCLUDE_FROM_ALL gtest/gtest_main.cc)
target_link_libraries(gtest_main gtest ${ANDROID_LOG_LIB})

set_target_properties(gtest gtest_main PROPERTIES CMAKE_CXX_CLANG_TIDY ""
                                                  CMAKE_CXX_INCLUDE_WHAT_YOU_USE "")

foreach(target gtest gtest_main)
  set_target_properties(${target} PROPERTIES CXX_CLANG_TIDY "")
endforeach()

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(vision_graph_test test_tensor_numeric.cpp)
target_link_libraries(vision_graph_test gtest gtest_main ${ANDROID_LOG_LIB})
target_link_libraries(vision_graph_test ${DLCVLIB} ${OpenCV_LIBRARIES} ${TBB_LIBS} dl ${ANDROID_LOG_LIB})

add_test(NAME vision_graph_test COMMAND vision_graph_test WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})