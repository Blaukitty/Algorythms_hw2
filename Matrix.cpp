// Matrix.cpp
#include "Matrix.hpp"
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>        
#include <iostream>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

using namespace std;

Matrix read(const string& path)
{
    ifstream f(path);
    if (!f) throw runtime_error("CSV not found: " + path);

    vector<double> values;
    size_t rows = 0, cols = 0;
    string line, cell;

    while (getline(f, line)) {
        ++rows;
        stringstream ss(line);
        while (getline(ss, cell, ',')) values.push_back(stod(cell));
        if (!cols) cols = values.size() / rows;
    }
    return Eigen::Map<Matrix>(values.data(), rows, cols);
}

void write(const string& path, const Vector& x)
{
    ofstream f(path);
    for (Eigen::Index i = 0; i < x.size(); ++i) f << x(i) << '\n';
}

Vector Guess(Matrix A)          // принимаем копию, чтобы можно было менять
{
    int n = augmented.rows();
    int m = augmented.cols();

    Vector b = augmented.col(m - 1);
    Matrix A = augmented.block(0, 0, n, n);
    
    for (int k = 0; k < n; ++k)
    {
        // частичный поиск главного элемента
        int p = A.col(k).segment(k, n - k).cwiseAbs().maxCoeff(&p);
        p += k;
        swap(A.row(k), A.row(p));
        swap(b(k), b(p));
        
        // 2.2 вектор коэффициентов
        Vector factor = A.col(k).segment(k+1, n-k-1) / A(k,k);

        // 2.3 обнуляем под главной диагональю
        A.block(k+1, k+1, n-k-1, n-k-1).noalias() -=
            factor * A.row(k).segment(k+1, n-k-1);
        b.segment(k+1, n-k-1).noalias() -= factor * b(k);
    }

    // обратный ход
    Vector x(n);
    for (int i = n - 1; i >= 0; --i) {
        double s = A.row(i).segment(i + 1, n - i - 1)
                         .dot(x.segment(i + 1, n - i - 1));
        x(i) = (b(i) - s) / A(i, i);
    }
    return x;
}


Matrix randomSystem(int n, unsigned seed)
{
    if (n <= 0 || n > 4'000)
        throw std::invalid_argument("n out of range (1‒4000)");

    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dist(-10.0, 10.0);

    Matrix A(n, n + 1);
    for (int i = 0; i < A.size(); ++i) A(i) = dist(gen);
    A.topLeftCorner(n, n).diagonal().array() += n;   // невырожденность
    return A;
}
