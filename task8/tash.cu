#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace opt = boost::program_options;

template <class ctype>
class Data {
private:
    int len;
    ctype* d_arr;
public:
    std::vector<ctype> arr;
    Data(int length) : len(length), arr(len), d_arr(nullptr) {
        cudaError_t err = cudaMalloc((void**)&d_arr, len * sizeof(ctype));
        if (err != cudaSuccess) {
            std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    ~Data() {
        if (d_arr) {
            cudaFree(d_arr);
        }
    }
    void copyToDevice() {
        cudaError_t err = cudaMemcpy(d_arr, arr.data(), len * sizeof(ctype), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "CUDA memory copy to device failed: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    void copyToHost() {
        cudaError_t err = cudaMemcpy(arr.data(), d_arr, len * sizeof(ctype), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "CUDA memory copy to host failed: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    ctype* getDevicePointer() {
        return d_arr;
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

__global__ void iterate(double* matrix, double* lastMatrix, int size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j == 0 || i == 0 || i >= size - 1 || j >= size - 1) return;

    matrix[i * size + j] = 0.25 * (lastMatrix[i * size + j + 1] + lastMatrix[i * size + j - 1] +
                                   lastMatrix[(i - 1) * size + j] + lastMatrix[(i + 1) * size + j]);
}

template <unsigned int blockSize>
__global__ void compute_error(double* matrix, double* lastMatrix, double* max_error, int size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= size || i >= size) return;

    __shared__ typename cub::BlockReduce<double, blockSize>::TempStorage temp_storage;
    double local_max = 0.0;

    if (j > 0 && i > 0 && j < size - 1 && i < size - 1) {
        int index = i * size + j;
        double error = fabs(matrix[index] - lastMatrix[index]);
        local_max = error;
    }

    double block_max = cub::BlockReduce<double, blockSize>(temp_storage).Reduce(local_max, cub::Max());

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicMax(reinterpret_cast<unsigned long long*>(max_error), __double_as_longlong(block_max));
    }
}

struct CudaGraphDeleter {
    void operator()(cudaGraph_t* graph) const {
        if (graph) {
            cudaGraphDestroy(*graph);
            delete graph;
        }
    }
};

struct CudaGraphExecDeleter {
    void operator()(cudaGraphExec_t* graphExec) const {
        if (graphExec) {
            cudaGraphExecDestroy(*graphExec);
            delete graphExec;
        }
    }
};

struct CudaStreamDeleter {
    void operator()(cudaStream_t* stream) const {
        if (stream) {
            cudaStreamDestroy(*stream);
            delete stream;
        }
    }
};


void savematrix(const double* matrix, int size, const std::string& filename) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    int fieldWidth = 10;

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            outputFile << std::setw(fieldWidth) << std::fixed << std::setprecision(4) << matrix[i * size + j];
        }
        outputFile << std::endl;
    }

    outputFile.close();
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
    double error;
    error = accuracy + 1;
    int iter = 0;

    dim3 blockDim(32, 32);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x, (size + blockDim.y - 1) / blockDim.y);
    Data<double> d_max_error(gridDim.x * gridDim.y);
    d_max_error.copyToDevice();
    matrix.copyToDevice();
    lastMatrix.copyToDevice();
    double* matrix_link = matrix.getDevicePointer();
    double* lastMatrix_link = lastMatrix.getDevicePointer();
    double* d_max_error_link = d_max_error.getDevicePointer();

    std::unique_ptr<cudaStream_t, CudaStreamDeleter> stream(new cudaStream_t);
    std::unique_ptr<cudaGraph_t, CudaGraphDeleter> graph(new cudaGraph_t);
    std::unique_ptr<cudaGraphExec_t, CudaGraphExecDeleter> graphExec(new cudaGraphExec_t);

    cudaStreamCreate(stream.get());

    bool graphCreated = false;


cudaMemset(d_max_error_link, 0, sizeof(double));

while (iter < countIter && error > accuracy) {
    if (!graphCreated) {
        cudaStreamBeginCapture(*stream, cudaStreamCaptureModeGlobal);

        for(int i = 0; i < 999; i++){
            iterate<<<gridDim, blockDim, 0, *stream>>>(matrix_link, lastMatrix_link, size);
            std::swap(lastMatrix_link, matrix_link);
        }

        iterate<<<gridDim, blockDim, 0, *stream>>>(matrix_link, lastMatrix_link, size);
        compute_error<32><<<gridDim, blockDim, 0, *stream>>>(matrix_link, lastMatrix_link, d_max_error_link, size);

        cudaStreamEndCapture(*stream, graph.get());
        cudaGraphInstantiate(graphExec.get(), *graph, nullptr, nullptr, 0);

        graphCreated = true;
    } else {
        cudaGraphLaunch(*graphExec, *stream);
        cudaStreamSynchronize(*stream);

        double temp_error;
        cudaMemcpy(&temp_error, d_max_error_link, sizeof(double), cudaMemcpyDeviceToHost);
        error = temp_error;

        std::cout << "Iteration: " << iter + 1000 << ", Error: " << error << std::endl;

        iter += 1000;
        cudaMemset(d_max_error_link, 0, sizeof(double));
    }
}

    if (size <= 13) {
        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                std::cout << matrix.arr[i * size + j] << ' ';
            }
            std::cout << std::endl;
        }
    }

    matrix.copyToHost();
    auto end = std::chrono::high_resolution_clock::now();
    auto time_s = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time: " << time_s << " milliseconds." << std::endl;
    savematrix(matrix.arr.data(), size , "cuda.txt");

    return 0;
}
