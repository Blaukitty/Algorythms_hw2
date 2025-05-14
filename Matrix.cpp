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

Vector Guess(Matrix M) {
    // M — это расширенная матрица n×(n+1)
    int n = M.rows();
    // последний столбец — вектор свободных членов
    Vector b = M.col(n);
    // первые n столбцов — сама квадратная матрица
    M.conservativeResize(n, n);

    // Прямой ход
    for(int k = 0; k < n; ++k) {
        // выбираем строку с максимальным элементом в столбце k
        int rel_max;
        M.col(k).segment(k, n-k)
            .cwiseAbs()
            .maxCoeff(&rel_max);
        int p = k + rel_max;
        // меняем местами строки k и p
        M.row(k).swap(M.row(p));
        std::swap(b(k), b(p));

        double pivot = M(k,k);
        if(std::abs(pivot) < 1e-12)
            throw std::runtime_error("Singular matrix!");

        // убираем элемент под диагональю
        for(int i = k+1; i < n; ++i) {
            double f = M(i,k) / pivot;
            // вычитаем f * строка_k из строки_i (коэфф-ты)
            M.row(i).segment(k, n-k) 
              -= f * M.row(k).segment(k, n-k);
            // и тот же f из b
            b(i) -= f * b(k);
        }
    }

    // Обратный ход
    Vector x(n);
    for(int i = n-1; i >= 0; --i) {
        double sum = 0;
        for(int j = i+1; j < n; ++j) {
            sum += M(i,j) * x(j);
        }
        x(i) = (b(i) - sum) / M(i,i);
    }

    return x;
}

Matrix randomSystem(int n, unsigned seed)
{
    if (n <= 0 || n > 4'000)
        throw std::invalid_argument("n out of range (1‒4000)");

    mt19937 gen(seed);
    uniform_real_distribution<> dist(-10, 10);

    Matrix M(n, n + 1);
    for (int i = 0; i < M.size(); ++i) M(i) = dist(gen);
    M.topLeftCorner(n, n).diagonal().array() += n;   // невырожденность
    return M;
}
