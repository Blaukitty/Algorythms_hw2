// test_gauss.cpp
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "Matrix.cpp"          
#include <Eigen/Dense>

TEST_CASE("3×3 hand example") {
    Eigen::Matrix A(3,4);
    A << 2,1,-1,  8,
         -3,-1,2, -11,
         -2,1,2, -3;
    REQUIRE(Guess(A).isApprox(
        (Eigen::Vector3d()<<2,-3,1).finished()
    ));
}

TEST_CASE("Compare with Eigen solver") {
    int n = 100;
    auto sys = randomSystem(n, 123);
    auto x = Guess(sys);
    auto A = sys.block(0,0,n,n);
    auto b = sys.col(n);
    REQUIRE(x.isApprox(A.fullPivLu().solve(b), 1e-8));
}
