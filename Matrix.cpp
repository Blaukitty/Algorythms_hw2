#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <random>
#include <iostream>

using Matrix = Eigen::MatrixXd;  
using Vector = Eigen::VectorXd;
using namespace std;

Matrix read(const string& p) {

	ifstream f(p);
	if (!f) {
		throw runtime_error("CSV not found");
	}

	vector<double> values;
	size_t rows = 0, cols = 0;

	string line;
	while (getline(f, line)) {
		++rows;
		stringstream ss(line); string cell;

		while (getline(ss, cell, ',')) {
			values.push_back(stod(cell));
		}
		if (!cols) {
			cols = values.size();
		}
	}
	return Eigen::Map<M>(values.data(), rows, cols);
}

void write(const string& p, const Vector& x) {
	ofstream f(p);
	for (int i = 0; i < x.size(); ++i) {
		f << x(i) << '\n';
	}
}

Vector Guess(const Matrix& A) { // сам метод Гауса
	const int n = A.rows();
	VectorXd b = A.col(n - 1);          // последний столбец
	A.conservativeResize(n, n - 1);     // отбросили b из A

	for (int k = 0; k < n; ++k) {
		// поиск главного элемента
		int p = (A.col(k).segment(k, n - k)).cwiseAbs().maxCoeff(&p) + k;
		if (std::abs(A(p, k)) < 1e-12) throw std::runtime_error("Singular!");

		// перестановка строк
		A.row(k).swap(A.row(p));
		swap(b(k), b(p));

		// вектор коэффициентов
		Vector factor = A.col(k).segment(k + 1, n - k - 1) / A(k, k);

		// обнуляем под диагональю
		A.block(k + 1, k, n - k - 1, n - k)   // включая текущий столбец
			.rowwise()                  //   ― удобная альтернатива, но:
			-= factor * A.row(k);       //   используем блок аналогично b

		b.segment(k + 1, n - k - 1).noalias() -= factor * b(k);

		Vector x(n);   //в обратную сторону
		for (int i = n - 1; i >= 0; --i) {
			double s = A.row(i).segment(i + 1, n - i - 1).dot(x.segment(i + 1, n - i - 1));
			x(i) = (b(i) - s) / A(i, i);
		}
		return x;
}

Matrix randomSystem(int n, unsigned seed = 42) {
    mt19937 gen(seed);
    uniform_real_distribution<> dist(-10.0, 10.0);
    MatrixXd A(n, n+1);
    for (int i=0;i<A.size();++i) A(i) = dist(gen);
    // гарантируем невырожденность добавив n к диагонали
    A.topLeftCorner(n,n).diagonal().array() += n;
    return A;
} 

int main(int a, char** a) {
	if (a < 3) {
		cerr << "Usage: solver in.csv out.csv\n";
		return 1;
	}
	Matrix aug = read(a[1]);
	Vector x = Guess(aug);
	write(a[2], x);
	cout << "Solved: see " << a[2] << '\n';
}
	
