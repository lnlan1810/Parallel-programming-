#include <iostream>
#include <mpi.h>
#include <cstdlib>
#include <ctime>

using namespace std;

int main(int argc, char** argv) {
    int rank, size, n;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        cout << "Enter the size of the matrix: ";
        cin >> n;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double* A = new double[n * n];
    double* b = new double[n];
    double* x = new double[n];

    if (rank == 0) {
        srand(time(NULL));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i * n + j] = rand() % 10 + 1;
            }
            b[i] = rand() % 10 + 1;
        }
    }

    start_time = MPI_Wtime();

    MPI_Bcast(A, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int k = 0; k < n - 1; k++) {
        if (k % size == rank) {
            for (int i = k + 1; i < n; i++) {
                double factor = A[i * n + k] / A[k * n + k];
                for (int j = k + 1; j < n; j++) {
                    A[i * n + j] -= factor * A[k * n + j];
                }
                b[i] -= factor * b[k];
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        for (int i = k + 1; i < n; i++) {
            if (i % size == rank) {
                double factor = A[i * n + k] / A[k * n + k];
                for (int j = k + 1; j < n; j++) {
                    A[i * n + j] -= factor * A[k * n + j];
                }
                b[i] -= factor * b[k];
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0;
            for (int j = i + 1; j < n; j++) {
                sum += A[i * n + j] * x[j];
            }
            x[i] = (b[i] - sum) / A[i * n + i];
        }
    }

    end_time = MPI_Wtime();

    if (rank == 0) {
        cout << "Time taken: " << end_time - start_time << " seconds" << endl;
    }

    delete[] A;
    delete[] b;
    delete[] x;

    MPI_Finalize();

    return 0;
}