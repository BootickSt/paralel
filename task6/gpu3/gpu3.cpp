 
#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>
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

void init(std::vector<double>& arr, int size) {
    arr[0] = 10.0;
    arr[size-1] = 20.0;
    arr[(size-1)*size + (size-1)] = 30.0;
    arr[(size-1)*size] = 20.0;

    for (size_t i = 1; i < size-1; i++) {
        arr[0*size+i] = linearInterpolation(i, 0.0, arr[0], size-1, arr[size-1]);
        arr[i*size+0] = linearInterpolation(i, 0.0, arr[0], size-1, arr[(size-1)*size]);
        arr[i*size+(size-1)] = linearInterpolation(i, 0.0, arr[size-1], size-1, arr[(size-1)*size + (size-1)]);
        arr[(size-1)*size+i] = linearInterpolation(i, 0.0, arr[(size-1)*size], size-1, arr[(size-1)*size + (size-1)]);
    }
}

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
        ("accuracy",opt::value<double>(),"Accuracy")
        ("size",opt::value<int>(),"Size of matrix")
        ("count",opt::value<int>(),"Count of iteretions")
        ("help","Help");

    opt::variables_map vm;

    opt::store(opt::parse_command_line(argc, argv, desc), vm);

    opt::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    int size = vm["size"].as<int>();
    double accuracy = vm["accuracy"].as<double>();
    int count_iter = vm["count"].as<int>();

    double error = 1.0;
    int iter = 0;

    Data<double> A(size * size);
    Data<double> Anew(size * size);

    init(A.arr, size);
    init(Anew.arr, size);

    auto start = std::chrono::high_resolution_clock::now();
    double* matrix = A.arr.data();
    double* last_matrix = Anew.arr.data();

    #pragma acc data copyin(error,last_matrix[0:size*size],matrix[0:size*size])
    {
        while (iter < count_iter && error > accuracy) {

            #pragma acc parallel loop collapse(2) present(matrix,last_matrix)

            for (size_t i = 1; i < size-1; i++) {
                for (size_t j = 1; j < size-1; j++) {

                    matrix[i*size+j]  = 0.25 * (last_matrix[i*size+j+1] + last_matrix[i*size+j-1] + last_matrix[(i-1)*size+j] + last_matrix[(i+1)*size+j]);
                }
            }

            if ((iter+1)%1000 == 0){
                error = 0.0;
                #pragma acc update device(error)
                #pragma acc parallel loop collapse(2) reduction(max:error) present(matrix,last_matrix)

                for (size_t i = 1; i < size-1; i++) {
                    for (size_t j = 1; j < size-1; j++) {
                        error = fmax(error,fabs(matrix[i*size+j]-last_matrix[i*size+j]));
                    }
                }
                #pragma acc update self(error)
            // std::cout << "iter № " << iter << " Error = " << error << std::endl;
            }

            std::swap(last_matrix, matrix);
            iter++;
        }
        #pragma acc update self(matrix[0:size*size])
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    auto time_s = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

   std::cout<<"time = "<< time_s << "ms"<< std::endl;

    if (size <= 13) {
        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                std::cout << A.arr[i * size + j] << ' ';
            }
            std::cout << std::endl;
        }
    }

    savematrix(A.arr.data(), size , "matrix.txt");

    return 0;
}








