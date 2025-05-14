#include "Matrix.hpp"
#include <iostream>

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: solver in.csv out.csv\n";
        return 1;
    }
    Matrix aug = read(argv[1]);
    Vector x   = Guess(aug);
    write(argv[2], x);
}
