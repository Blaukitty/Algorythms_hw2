#pragma once
#include <Eigen/Dense>
#include <string>
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

Matrix  read (const std::string& path);
void    write(const std::string& path, const Vector& x);
Vector  Guess(const Matrix A);
Matrix  randomSystem(int n, unsigned seed = 42);
