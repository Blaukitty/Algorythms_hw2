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
    string line;
    while (getline(f, line)) {
        ++rows;
        stringstream ss(line);
        string cell;
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

Vector Guess(const Matrix& aug)          
{
    int n = A.rows();  
    Vector b = A.col(n);

    A.conservativeResize(n, n);
    
    for (int k = 0; k < n; ++k)
    {
        int p;
        A.col(k).segment(k, n - k).cwiseAbs().maxCoeff(&p);
        p += k;

        if (std::abs(A(p, k)) < 1e-12)
            throw std::runtime_error("Singular!");
        
        A.row(k).swap(A.row(p)); 
        swap(b(k), b(p));
        
        // вектор коэффициентов
        Vector factor = A.col(k).segment(k+1, n-k-1) / A(k,k);

        // обнуляем под главной диагональю
        A.block(k+1, k, n-k-1, n-k)
         .noalias() -= factor * A.row(k).segment(k, n-k);
        b.segment(k+1, n-k-1)
         .noalias() -= factor * b(k);
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
    std::uniform_real_distribution<> dist(-10, 10);

    Matrix M(n, n + 1);
    for (int i = 0; i < M.size(); ++i) M(i) = dist(gen);
    M.topLeftCorner(n, n).diagonal().array() += n;   // невырожденность
    return M;
}
