#include <iostream>
#include <iomanip>
#include <omp.h>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <boost/program_options.hpp>
#include <vector>
#include <chrono>

namespace opt = boost::program_options;

template <class ctype>
class Data {
private:
    int len;
public:
    std::vector<ctype> arr;
    Data(int length) : len(length), arr(len) {
        #pragma acc enter data copyin(this)
        #pragma acc enter data create(arr[0:len])
    }
    ~Data() {
        #pragma acc exit data delete(arr)
        #pragma acc exit data delete(this)
    }
};

double linearInterpolation(double x, double x1, double y1, double x2, double y2) {
    return y1 + ((x - x1) * (y2 - y1) / (x2 - x1));
}

void init(Data<double>& matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix.arr[i * size + j] = 0;
        }
    }
    matrix.arr[0] = 10.0;
    matrix.arr[size - 1] = 20.0;
    matrix.arr[(size - 1) * size + (size - 1)] = 30.0;
    matrix.arr[(size - 1) * size] = 20.0;
    for (int i = 1; i < size - 1; ++i) {
        matrix.arr[i * size + 0] = linearInterpolation(i, 0.0, matrix.arr[0], size - 1, matrix.arr[(size - 1) * size]);
    }
    for (int i = 1; i < size - 1; ++i) {
        matrix.arr[0 * size + i] = linearInterpolation(i, 0.0, matrix.arr[0], size - 1, matrix.arr[size - 1]);
    }
    for (int i = 1; i < size - 1; ++i) {
        matrix.arr[(size - 1) * size + i] = linearInterpolation(i, 0.0, matrix.arr[(size - 1) * size], size - 1, matrix.arr[(size - 1) * size + (size - 1)]);
    }
    for (int i = 1; i < size - 1; ++i) {
        matrix.arr[i * size + (size - 1)] = linearInterpolation(i, 0.0, matrix.arr[size - 1], size - 1, matrix.arr[(size - 1) * size + (size - 1)]);
    }
}

int main(int argc, char const *argv[]) {
    opt::options_description desc("options");
    desc.add_options()
        ("accuracy", opt::value<double>(), "Accuracy")
        ("size", opt::value<int>(), "Size of matrix")
        ("count", opt::value<int>(), "Count of iterations")
        ("help", "Help");
    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, desc), vm);
    opt::notify(vm);
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }
    if (!vm.count("size") || !vm.count("accuracy") || !vm.count("count")) {
        std::cerr << "Missing required arguments: size, accuracy, or count.\n";
        return 1;
    }
    auto start = std::chrono::high_resolution_clock::now();
    int size = vm["size"].as<int>();
    double accuracy = vm["accuracy"].as<double>();
    int countIter = vm["count"].as<int>();

    Data<double> matrix(size * size);
    Data<double> lastMatrix(size * size);
    init(matrix, size);
    init(lastMatrix, size);
    double* matrix1 = matrix.arr.data();
    double* lastMatrix1 = lastMatrix.arr.data();
    double error;

    error = accuracy + 1;

    int iter = 0;
    const double coef = -1.0;
    int idx;
    cublasHandle_t handle;
    cublasCreate(&handle);

    #pragma acc data copyin(matrix1[0:size*size], lastMatrix1[0:size*size]) copy(error)
    {
        while (iter < countIter) {
            #pragma acc parallel loop independent collapse(2) present(matrix1, lastMatrix1)
            for (size_t i = 1; i < size - 1; i++) {
                for (size_t j = 1; j < size - 1; j++) {
                    matrix1[i * size + j] = 0.25 * (lastMatrix1[i * size + j + 1] + 
                                                    lastMatrix1[i * size + j - 1] + 
                                                    lastMatrix1[(i - 1) * size + j] + 
                                                    lastMatrix1[(i + 1) * size + j]);
                }
            }

            if ((iter + 1) % 1000 == 0) {
                #pragma acc host_data use_device(matrix1, lastMatrix1, error)
                {
                    cublasDaxpy(handle, size * size, &coef, matrix1, 1, lastMatrix1, 1);
                    cublasIdamax(handle, size * size, lastMatrix1, 1, &idx);
                }
                #pragma acc parallel present(lastMatrix1, error)
                {
                    error = fabs(lastMatrix1[idx - 1]);
                }
                #pragma acc update self(error)
                if (error <= accuracy) break;
                
                #pragma acc host_data use_device(matrix1, lastMatrix1)
                {
                    cublasDcopy(handle, size * size, matrix1, 1, lastMatrix1, 1);
                }
            } 
            else {
                {
                    std::swap(matrix1, lastMatrix1);
                }
            }
            iter++;
        }
    }
    #pragma acc exit data delete(error)
    cublasDestroy(handle);
    auto end = std::chrono::high_resolution_clock::now();
    auto time_s = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time: " << time_s << " milliseconds." << std::endl;

    return 0;
}
