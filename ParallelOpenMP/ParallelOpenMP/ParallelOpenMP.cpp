#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <omp.h>

void RandomDataInitialization(std::vector<double>& matrix, std::vector<double>& vector, int size) {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    for (int i = 0; i < size; ++i) {
        vector[i] = std::rand() / static_cast<double>(RAND_MAX);
        for (int j = 0; j < size; ++j) {
            matrix[i * size + j] = std::rand() / static_cast<double>(RAND_MAX);
        }
    }
}

void GaussianElimination(std::vector<double>& matrix, std::vector<double>& vector, int size) {
#pragma omp parallel for
    for (int k = 0; k < size - 1; ++k) {
        for (int i = k + 1; i < size; ++i) {
            double factor = matrix[i * size + k] / matrix[k * size + k];
            for (int j = k; j < size; ++j) {
                matrix[i * size + j] -= factor * matrix[k * size + j];
            }
            vector[i] -= factor * vector[k];
        }
    }
}

void BackSubstitution(const std::vector<double>& matrix, const std::vector<double>& vector,
    std::vector<double>& result, int size) {
    for (int i = size - 1; i >= 0; --i) {
        result[i] = vector[i] / matrix[i * size + i];
        for (int j = i - 1; j >= 0; --j) {
            result[j] -= matrix[j * size + i] * result[i];
        }
    }
}

int main() {
    int size;

    std::cout << "Enter the size of the matrix: ";
    std::cin >> size;

    std::vector<double> matrix(size * size);
    std::vector<double> vector(size);
    std::vector<double> result(size);

    RandomDataInitialization(matrix, vector, size);

    double start_time = omp_get_wtime();

    GaussianElimination(matrix, vector, size);
    BackSubstitution(matrix, vector, result, size);

    double end_time = omp_get_wtime();
    double execution_time = end_time - start_time;

    std::cout << "Result Vector: ";
    for (int i = 0; i < size; ++i) {
        std::cout << std::setw(8) << std::fixed << std::setprecision(4) << result[i] << " ";
    }

    std::cout << "\nExecution Time: " << execution_time << " seconds\n";

    return 0;
}