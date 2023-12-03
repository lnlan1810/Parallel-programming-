#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

void RandomDataInitialization(std::vector<double>& matrix, int size) {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i * size + j] = std::rand() / static_cast<double>(RAND_MAX);
        }
    }
}

void GaussianElimination(std::vector<double>& matrix, int size) {
    for (int k = 0; k < size - 1; ++k) {
        for (int i = k + 1; i < size; ++i) {
            double factor = matrix[i * size + k] / matrix[k * size + k];
            for (int j = k; j < size; ++j) {
                matrix[i * size + j] -= factor * matrix[k * size + j];
            }
        }
    }
}

int main() {
    int size;

    std::cout << "Enter the size of the matrix: ";
    std::cin >> size;

    std::vector<double> matrix(size * size);

    RandomDataInitialization(matrix, size);

    double start_time = omp_get_wtime();

    GaussianElimination(matrix, size);

    double end_time = omp_get_wtime();
    double execution_time = end_time - start_time;

    std::cout << "Execution Time: " << execution_time << " seconds\n";

    return 0;
}