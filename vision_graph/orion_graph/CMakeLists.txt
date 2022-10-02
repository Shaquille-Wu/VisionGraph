cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)
set(CMAKE_VERBOSE_MAKEFILE  ON)


if(TARGET_ARCH STREQUAL "x86_64")
     set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -march=x86-64")
elseif(TARGET_OS STREQUAL "android")
     set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DANDROID -fexceptions -frtti  -Wno-unused-private-field")
     if(TARGET_ARCH STREQUAL "armv8")
          set(ANDROID_ABI  "arm64-v8a")
     else()
          set(ANDROID_ABI  "armeabi-v7a")
     endif()
else()
     message(FATAL_ERROR "unsupported TARGET_OS ${TARGET_OS}, TARGET_ARCH ${TARGET_ARCH}")
endif()

#set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-unused-variable")
if (ANDROID_PLATFORM)
     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
endif()

set(TBB_LIBS TBB::tbb TBB::tbbmalloc_proxy TBB::tbbmalloc)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/vision_graph/include/")

file(GLOB_RECURSE COMMON_SRC            "${CMAKE_CURRENT_SOURCE_DIR}/vision_graph/common/*.cpp")
file(GLOB_RECURSE SOLVER_INTUITIVE_SRC  "${CMAKE_CURRENT_SOURCE_DIR}/vision_graph/solvers/intuitive_solvers/*.cpp")
file(GLOB_RECURSE SOLVER_FUNCTION_SRC   "${CMAKE_CURRENT_SOURCE_DIR}/vision_graph/solvers/function_solvers/*.cpp")
file(GLOB         SOLVER_SRC            "${CMAKE_CURRENT_SOURCE_DIR}/vision_graph/solvers/*.cpp")
file(GLOB         GRAPH_DEF_SRC         "${CMAKE_CURRENT_SOURCE_DIR}/vision_graph/*.cpp")

set (GRAPH_SRC 
     ${COMMON_SRC}
     ${SOLVER_INTUITIVE_SRC}
     ${SOLVER_FUNCTION_SRC}
     ${SOLVER_SRC}
     ${GRAPH_DEF_SRC}
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/dlcv/include)
set(DLCVLIB ${CMAKE_CURRENT_SOURCE_DIR}/deps/dlcv/libs/${TARGET_ARCH}-${TARGET_OS}/libdlcv.a)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/fastcv/include)
set(FASTCVLIB ${CMAKE_CURRENT_SOURCE_DIR}/deps/fastcv/libs/libfastcv.a)

if(TARGET_OS STREQUAL "android" AND OpenCV_DIR STREQUAL "")
     set(OpenCV_INCLUDE_DIRS    "${CMAKE_CURRENT_SOURCE_DIR}/deps/opencv/include")
     set(OpenCV_LIB_DIRS        "${CMAKE_CURRENT_SOURCE_DIR}/deps/opencv/lib/${CMAKE_ANDROID_ARCH_ABI}")
     file(GLOB OpenCV_LIBRARIES "${OpenCV_LIB_DIRS}/*.so")  ##some libs are *.so, and some libs are *.a, so, we should change it according to real case
else()
     find_package(OpenCV REQUIRED)
endif()
include_directories(
      ${OpenCV_INCLUDE_DIRS}
)  

set(single_face_tracker "${CMAKE_CURRENT_SOURCE_DIR}/vision_graph/solvers/function_solvers/implement/tracker/single_face_tracker/eco_tracker/include")
include_directories(${single_face_tracker})

set(apce_conf "${CMAKE_CURRENT_SOURCE_DIR}/vision_graph/solvers/function_solvers/implement/apce_conf")
include_directories(${apce_conf})

set(face_alignment "${CMAKE_CURRENT_SOURCE_DIR}/vision_graph/solvers/function_solvers/implement/face_alignment")
include_directories(${face_alignment})

set(fhog "${CMAKE_CURRENT_SOURCE_DIR}/vision_graph/solvers/function_solvers/implement/fhog")
include_directories(${fhog})

set(sideface_cls "${CMAKE_CURRENT_SOURCE_DIR}/vision_graph/solvers/function_solvers/implement/tracker/single_face_tracker/sideface_cls/include")
include_directories(${sideface_cls})

set(strategy "${CMAKE_CURRENT_SOURCE_DIR}/vision_graph/solvers/function_solvers/implement/tracker/single_face_tracker/strategy/include")
include_directories(${strategy})

set(strategy_utils "${CMAKE_CURRENT_SOURCE_DIR}/vision_graph/solvers/function_solvers/implement/tracker/single_face_tracker/strategy/include/utils/include")
include_directories(${strategy_utils})

set(strategy_base "${CMAKE_CURRENT_SOURCE_DIR}/vision_graph/solvers/function_solvers/implement/tracker/single_face_tracker/strategy/include/base/include")
include_directories(${strategy_base})

set(implement "${CMAKE_CURRENT_SOURCE_DIR}/vision_graph/solvers/function_solvers/implement")
include_directories(${implement})


set(eigen3 "${CMAKE_CURRENT_SOURCE_DIR}/vision_graph/common/eigen3/Eigen")
include_directories(${eigen3})

add_library(vision_graph SHARED ${GRAPH_SRC} )
set(GRAPH_DEP_LIBS ${DLCVLIB} ${OpenCV_LIBRARIES} dl)
if(TARGET_OS STREQUAL "android")
     set(ANDROID_LOG_LIB log)
     set(GRAPH_DEP_LIBS ${GRAPH_DEP_LIBS} ${FASTCVLIB} ${ANDROID_LOG_LIB})
     add_definitions(-DFASTCV_X)
endif()
target_link_libraries(vision_graph ${TBB_LIBS} ${GRAPH_DEP_LIBS}
        z
        m)

if(TARGET_ARCH STREQUAL "x86_64")
     add_subdirectory(tools)
endif()
add_subdirectory(examples)
add_subdirectory(tests)
add_subdirectory(demos)

#file(GLOB TBB_LIB               "${PROJECT_BUILD_ROOT}/${TBB_CMAKE_BUILD_DIR}/${TBB_CMAKE_BUILD_DIR}${TBB_GENERATE_CFG}/libtbb*${TBB_GENERATE_SUFFIX}.so*")
install (TARGETS   vision_graph DESTINATION lib)
install (DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/vision_graph/include DESTINATION .)