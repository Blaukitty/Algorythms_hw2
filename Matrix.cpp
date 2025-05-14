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

Matrix read(const std::string& path)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("CSV not found: " + path);

    std::vector<double> values;
    std::size_t rows = 0, cols = 0;
    std::string line, cell;

    while (std::getline(f, line)) {
        ++rows;
        std::stringstream ss(line);
        while (std::getline(ss, cell, ',')) values.push_back(std::stod(cell));
        if (!cols) cols = values.size() / rows;
    }
    return Eigen::Map<Matrix>(values.data(), rows, cols);
}

void write(const std::string& path, const Vector& x)
{
    std::ofstream f(path);
    for (Eigen::Index i = 0; i < x.size(); ++i) f << x(i) << '\n';
}

Vector Guess(Matrix A)          // принимаем копию, чтобы можно было менять
{
    const int n = A.rows();
    Vector b = A.rightCols<1>();
    A = A.leftCols(n);         // отбросили правый столбец из A

    for (int k = 0; k < n; ++k)
    {
        // поиск главного элемента
        int p;
        A.col(k).segment(k, n - k).cwiseAbs().maxCoeff(&p);
        p += k;
        if (std::abs(A(p, k)) < 1e-12) throw std::runtime_error("Singular!");

        // перестановка
        A.row(k).swap(A.row(p));
        std::swap(b(k), b(p));

        // коэффициенты и зануление ниже диагонали
        Vector factor = A.col(k).segment(k + 1, n - k - 1) / A(k, k);
        A.block(k + 1, k, n - k - 1, n - k).noalias()
            -= factor * A.row(k).segment(k, n - k);
        b.segment(k + 1, n - k - 1).noalias() -= factor * b(k);
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
