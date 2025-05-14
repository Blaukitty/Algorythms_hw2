cmake_minimum_required(VERSION 3.15)
project(GaussSolver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Catch2 3 REQUIRED)         

add_executable(solver
    Matrix.cpp                           
)
target_link_libraries(solver PRIVATE Eigen3::Eigen)


enable_testing()

add_executable(gauss_tests
    tests_gauss.cpp                     
)
target_link_libraries(gauss_tests
    PRIVATE
        Eigen3::Eigen
        Catch2::Catch2                   
)

add_test(NAME GaussTests COMMAND gauss_tests)
