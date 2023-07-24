cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(linreg CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

include(${PROJECT_SOURCE_DIR}/cmake/conan.cmake)
conan_cmake_run(
  CONANFILE conanfile.txt
  BASIC_SETUP
  CMAKE_TARGETS # individual targets to link to
  BUILD
    missing 
)

set(CMAKE_CXX_STANDARD 17)

file(GLOB HRDS
    include/*.h
)
file(GLOB SRCS
    src/*.cpp
)

add_library(${PROJECT_NAME}
    ${SRCS}
    ${HRDS}
)

target_include_directories(${PROJECT_NAME} PUBLIC
    include
)

set(DEMO_COMMON_HDRS
    demo/helpers.h
)

set(DEMO_COMMON_SRCS
    demo/helpers.cpp
)

add_executable(demo_mean
    demo/demo_mean.cpp
    ${DEMO_COMMON_SRCS}
    ${DEMO_COMMON_HDRS}
)

target_include_directories(demo_mean PUBLIC
    include
)

target_link_libraries(demo_mean
    ${PROJECT_NAME}
)

add_executable(demo_linreg
    demo/demo_linreg.cpp
    ${DEMO_COMMON_SRCS}
    ${DEMO_COMMON_HDRS}
)

target_include_directories(demo_linreg PUBLIC
    include
)

target_link_libraries(demo_linreg
    ${PROJECT_NAME}
)


file(GLOB TEST_SRCS
    tests/*.cpp
)

add_executable(test_${PROJECT_NAME}
    ${TEST_SRCS}
    ${DEMO_COMMON_SRCS}
    ${DEMO_COMMON_HDRS}
)

add_custom_target(linreg_test
    $<TARGET_FILE:test_${PROJECT_NAME}>
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(test_${PROJECT_NAME} PUBLIC
    include
    demo
)

target_link_libraries(test_${PROJECT_NAME}
    CONAN_PKG::gtest
    ${PROJECT_NAME}
)