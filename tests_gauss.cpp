#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include "Matrix.hpp"          
//#include <Eigen/Dense>

TEST_CASE("3×3 hand example") {
    Matrix A(3,4);
    A <<  2, 1,-1,  8,
         -3,-1, 2,-11,
         -2, 1, 2, -3;
    Vector expected(3); 
    expected << 2,-3,1;

    REQUIRE( Guess(A).isApprox(expected) );
}

TEST_CASE("Compare with Eigen solver") {
    int n = 100;
    auto sys = randomSystem(n, 123);
    Vector x = Guess(sys);
    Matrix M = sys.block(0,0,n,n);
    Vector b = sys.col(n);
    REQUIRE(x.isApprox(A.fullPivLu().solve(b), 1e-8));
}
