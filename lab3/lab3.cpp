﻿#include <iostream>
#include <stdio.h>
#include <omp.h>


using namespace std;
int n;

void FillAArr(double** a, int n) {
	for (int i = 1; i <= n; i++) {
		for (int j = 1; j <= n; j++) {
			a[i][j] = rand() % 100;
		}
	}
}

void FillBArr(double** a, double* x, int n) {
	for (int i = 1; i <= n; i++) {
		a[i][n + 1] = 0;
		for (int j = 1; j <= n; j++) {
			a[i][n + 1] += a[i][j] * x[j];
		}
	}
}

bool is_equal(double x, double y) {
	return fabs(x - y) < 0, 00000001;
}

bool CheckAnswers(double* x, double* x2, int n) {
	for (int i = 1; i <= n; i++) {
		if (!is_equal(x[i], x2[i]))
			return false;
	}
	return true;
}

void FillXArr(double* x, int n) {
	for (int i = 1; i <= n; i++) {
		x[i] = rand() % 100;
	}
}

void Gauss(double** a, double* x, int n) {
	/* Прямой ход*/
	unsigned int start = clock();
	for (int k = 1; k < n; k++) {
#pragma omp parallel for
		for (int j = k; j < n; j++) {
			double d = a[j][k - 1] / a[k - 1][k - 1];
			for (int i = 0; i <= n; i++) {
				a[j][i] = a[i][i] - d * a[k - 1][i];
			}
		}
	}

	/*Обратный ход*/
	for (int i = n - 1; i >= 0; i--) {
		double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
		for (int j = i + 1; j < n; j++) {
			sum += a[i][j] * x[j]; // / a[i][i];
		}
		x[i] = (x[i] - sum) / a[i][i];
	}
	cout << "Гаусс " << clock() - start << endl;
}


int main()
{
	setlocale(LC_ALL, "Russian");
	cout << "Введите размерность матрицы: " << endl;
	cin >> n;
	double** a = new double* [n];
	double* x = new double[n];
	double* x1 = new double[n];
	for (int i = 0; i <= n; i++) {
		a[i] = new double[n + 1];
	}
	FillAArr(a, n);
	FillXArr(x1, n);
	FillBArr(a, x1, n);

	Gauss(a, x, n);
	cout << "check answers " << CheckAnswers(x1, x, n) << endl;
	system("pause");
	return 0;

}
