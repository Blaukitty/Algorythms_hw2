cmake_minimum_required(VERSION 3.15)
project(GaussSolver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Catch2 3 REQUIRED)

add_library(gauss_lib Matrix.cpp)
target_include_directories(gauss_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gauss_lib
    PUBLIC Eigen3::Eigen
)

add_executable(solver solver_main.cpp)
target_link_libraries(solver PRIVATE gauss_lib)

enable_testing()
add_executable(gauss_tests tests_gauss.cpp)
target_link_libraries(gauss_tests
    PRIVATE
        gauss_lib
        Catch2::Catch2WithMain   
)
add_test(NAME GaussTests COMMAND gauss_tests)
