#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <memory>
#include <fstream>
#include <iomanip>
#include </opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include/nvtx3/nvToolsExt.h>
#include <omp.h>
namespace opt = boost::program_options;

template <class ctype> 
class Data
    {
      private:
        int len;
      public:
        ctype *arr;
        Data(int length){
            len = length;
            arr = new ctype[len];
            #pragma acc enter data copyin(this)
            #pragma acc enter data create(arr[0:len])
        }
        ~Data(){
            #pragma acc exit data delete(arr)
            #pragma acc exit data delete(this)
            delete arr;
            len = 0;
        }
    };

void savematrix(const Data<double>& matrix, int size, const std::string& filename) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Unable to open file " << filename << " for writing." << std::endl;
        return;
    }
    int fieldWidth = 13;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            outputFile << std::setw(fieldWidth) << std::fixed << std::setprecision(4) << matrix.arr[i * size + j];
        }
        outputFile << std::endl;
    }
    outputFile.close();
}

double linearInterpolation(double x, double x1, double y1, double x2, double y2) {
    return y1 + ((x - x1) * (y2 - y1) / (x2 - x1));
}
double Error(Data<double>& last_matrix,Data<double>& matrix, int size){
    double error = 0.0;
    #pragma acc enter data copyin(matrix.arr[0:size*size], last_matrix.arr[0:size*size])
    #pragma acc update device(matrix.arr[0:size*size], last_matrix.arr[0:size*size])
    #pragma acc parallel loop reduction(max:error)
    for (size_t i = 1; i < size-1; i++)
    {
        #pragma acc loop
        for (size_t j = 1; j < size-1; j++)
        {
            
            error = fmax(error,fabs(matrix.arr[i*size+j] - last_matrix.arr[i*size+j]));
        }
    }
    #pragma acc exit data delete(matrix.arr, last_matrix.arr)
    return error;
}

void Iteraton(Data<double>& last_matrix,Data<double>& matrix,int size){
    #pragma acc enter data copyin(matrix.arr[0:size*size], last_matrix.arr[0:size*size])
    #pragma acc update device(matrix.arr[0:size*size], last_matrix.arr[0:size*size])
    #pragma acc parallel loop 
    for (size_t i = 1; i < size-1; i++)
    {
        #pragma acc loop
        for (size_t j = 1; j < size-1; j++)
        {
            matrix.arr[i*size+j]  = 0.25 * (last_matrix.arr[i*size+j+1] + last_matrix.arr[i*size+j-1] + last_matrix.arr[(i-1)*size+j] + last_matrix.arr[(i+1)*size+j]); 
        }
    }
    #pragma acc update self(matrix.arr[0:size*size])
    #pragma acc exit data delete(matrix.arr, last_matrix.arr)

}
void init(Data<double>& matrix,int size){
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            matrix.arr[i*size + j] = 0;
        }
    }
    matrix.arr[0] = 10.0;
    matrix.arr[size-1] = 20.0;
    matrix.arr[(size-1)*size + (size-1)] = 30.0;
    matrix.arr[(size-1)*size] = 20.0;
      for (size_t i = 1; i < size-1; i++)
    {
        matrix.arr[i*size+0] = linearInterpolation(i,0.0,matrix.arr[0],size-1,matrix.arr[(size-1)*size]);
    }
    for (size_t i = 1; i < size-1; i++)
    {
        matrix.arr[0*size+i] = linearInterpolation(i,0.0,matrix.arr[0],size-1,matrix.arr[size-1]);
    }
    for (size_t i = 1; i < size-1; i++)
    {
        matrix.arr[(size-1)*size+i] = linearInterpolation(i,0.0,matrix.arr[(size-1)*size],size-1,matrix.arr[(size-1)*size + (size-1)]);
    }
    for (size_t i = 1; i < size-1; i++)
    {
        matrix.arr[i*size+(size-1)] = linearInterpolation(i,0.0,matrix.arr[size-1],size-1,matrix.arr[(size-1)*size + (size-1)]);
    }
   
}

void swap(Data<double>& matrix, Data<double>& last_matrix, int size){
    #pragma acc enter data copyin(matrix.arr[0:size*size])
    #pragma acc update device(matrix.arr[0:size*size])
    double* curData = matrix.arr;
    double* prevData = last_matrix.arr;
    std::copy(curData, curData + size*size, prevData);
    #pragma acc update device(last_matrix.arr[0:size*size])
    #pragma acc update self(last_matrix.arr[0:size*size])
    #pragma acc exit data delete(matrix.arr)
}

int main(int argc, char const *argv[]){
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
    double start = omp_get_wtime();
    int size = vm["size"].as<int>();
    double accuracy = vm["accuracy"].as<double>();
    int count_iter = vm["count"].as<int>();
    Data<double> matrix(size*size);
    init(std::ref(matrix),size);
    Data<double> last_matrix(size*size);
    swap(std::ref(matrix),std::ref(last_matrix),size);
    double error = 1.0;
    int iter = 0;
    while (count_iter>iter && error > accuracy)
    {
        Iteraton(std::ref(last_matrix),std::ref(matrix),size);
        if ((iter+1)%100==0){  
        error = Error(std::ref(last_matrix),std::ref(matrix),size);
        std::cout << "iter №: "<< iter+1 << ' ' << "Error = " <<std::setprecision(8)<< error << std::endl;
        }
        swap(std::ref(matrix),std::ref(last_matrix),size);
        iter++;
    }

    if (size<=13){

    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            std::cout << matrix.arr[i*size+j] << ' ';
        }
        std::cout << std::endl;
    }
    }
    double end = omp_get_wtime();
    savematrix(std::ref(matrix), size , "matr.txt");
    std::cout << (end - start)<<std::endl;
}
