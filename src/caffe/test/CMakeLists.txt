# The option allows to include in build only selected test files and exclude all others
# Usage example:
#  cmake -DBUILD_only_tests="common,net,blob,im2col_kernel"


add_library(gtest STATIC ${CMAKE_SOURCE_DIR}/src/gtest/gtest.h ${CMAKE_SOURCE_DIR}/src/gtest/gtest-all.cpp)
caffe_default_properties(gtest)
#target_include_directories(gtest PUBLIC ${Caffe_SRC_DIR})
#target_compile_definitions(gtest PUBLIC -DGTEST_USE_OWN_TR1_TUPLE)



file(GLOB test_hdrs    ${CMAKE_SOURCE_DIR}/include/caffe/test/test_*.h*)
file(GLOB test_srcs    ${CMAKE_SOURCE_DIR}/src/caffe/test/test_*.cpp)


set(BUILD_only_tests "" CACHE STRING "Blank or comma-separated list of test files to build without 'test_' prefix and extension")
caffe_leave_only_selected_tests(test_srcs ${BUILD_only_tests})
caffe_leave_only_selected_tests(test_cuda ${BUILD_only_tests})

# For 'make runtest' target we don't need to embed test data paths to
# source files, because test target is executed in source directory
# That's why the lines below are commented. TODO: remove them

# definition needed to include CMake generated files
#add_definitions(-DCMAKE_BUILD) 

# generates test_data/sample_data_list.txt.gen.cmake
#caffe_configure_testdatafile(test_data/sample_data_list.txt)

set(the_target test.testbin)
set(test_args --gtest_shuffle)

#if(HAVE_CUDA)
#  caffe_cuda_compile(test_cuda_objs ${test_cuda})
#  list(APPEND test_srcs ${test_cuda_objs} ${test_cuda})
#else()
  list(APPEND test_args --gtest_filter="-*GPU*")
#endif()

# ---[ Adding test target
add_executable(${the_target} EXCLUDE_FROM_ALL ${test_srcs} ${CMAKE_SOURCE_DIR}/src/gtest/gtest.h ${CMAKE_SOURCE_DIR}/src/gtest/gtest-all.cpp)
target_link_libraries(${the_target} ${Caffe_LINK})
caffe_default_properties(${the_target})
caffe_set_runtime_directory(${the_target} "${PROJECT_BINARY_DIR}/test")

# ---[ Adding runtest
add_custom_target(runtest COMMAND ${the_target} ${test_args}
                          WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
