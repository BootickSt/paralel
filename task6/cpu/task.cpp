#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
namespace opt = boost::program_options;

double linearInterpolation(double x, double x1, double y1, double x2, double y2) {
    return y1 + ((x - x1) * (y2 - y1) / (x2 - x1));
}
void saveMatrixToFile(const std::unique_ptr<double[]>& matrix, int N, const std::string& filename) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Unable to open file " << filename << " for writing." << std::endl;
        return;
    }
    int fieldWidth = 10; 
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            outputFile << std::setw(fieldWidth) << std::fixed << std::setprecision(4) << matrix[i * N + j];
        }
        outputFile << std::endl;
    }
    outputFile.close();
}
void initMatrix(std::unique_ptr<double[]> &arr ,int N){
        
    arr[0] = 10.0;
    arr[N-1] = 20.0;
    arr[(N-1)*N + (N-1)] = 30.0;
    arr[(N-1)*N] = 20.0;
    for (size_t i = 1; i < N-1; i++)
    {
        arr[0*N+i] = linearInterpolation(i,0.0,arr[0],N-1,arr[N-1]);
        arr[i*N+0] = linearInterpolation(i,0.0,arr[0],N-1,arr[(N-1)*N]);
        arr[i*N+(N-1)] = linearInterpolation(i,0.0,arr[N-1],N-1,arr[(N-1)*N + (N-1)]);
        arr[(N-1)*N+i] = linearInterpolation(i,0.0,arr[(N-1)*N],N-1,arr[(N-1)*N + (N-1)]);
    }
}

int main(int argc, char const *argv[])
{
    opt::options_description desc("Argument");
    desc.add_options()
        ("accuracy",opt::value<double>()->default_value(1e-6),"Accuracy")
        ("cellsCount",opt::value<int>()->default_value(256),"Matrix size")
        ("iterCount",opt::value<int>()->default_value(50),"Count of itteration")
        ("help","help");
    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, desc), vm);
    opt::notify(vm);
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }
    int N = vm["cellsCount"].as<int>();
    double accuracy = vm["accuracy"].as<double>();
    int countIter = vm["iterCount"].as<int>();
    double error = 1.0;
    int iter = 0;
    std::unique_ptr<double[]> Matr(new double[N*N]);
    std::unique_ptr<double[]> Matrnew(new double[N*N]);
    initMatrix(Matr,N);
    initMatrix(Matrnew,N);
    double* prevmatrix = Matrnew.get();
    double* curmatrix = Matr.get();
    auto start = std::chrono::high_resolution_clock::now();
    while (iter < countIter && iter<10000000 && error > accuracy){
            #pragma acc parallel loop independent collapse(2) vector vector_length(80) gang num_gangs(40)
            for (size_t i = 1; i < N-1; i++)
            {
                for (size_t j = 1; j < N-1; j++)
                {
                    curmatrix[i*N+j]  = 0.25 * (prevmatrix[i*N+j+1] + prevmatrix[i*N+j-1] + prevmatrix[(i-1)*N+j] + prevmatrix[(i+1)*N+j]);
                }
            }
            if ((iter+1) %100 == 0)
            {
                error = 0.0;
                #pragma acc parallel loop independent collapse(2) reduction(max:error) gang num_gangs(40) vector vector_length(80) 
                for (size_t i = 1; i < N-1; i++)
                {   
                    for (size_t j = 1; j < N-1; j++)
                    {
                        error = fmax(error,fabs(curmatrix[i*N+j]-prevmatrix[i*N+j]));
                    }
                }
                std::cout << "iteration: " << iter+1 << ' ' << "error: " << error << std::endl;
            }
            double* temp = prevmatrix;
            prevmatrix = curmatrix;
            curmatrix = temp;
        iter++;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto time_s = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); 
                std::cout<< "time: " << time_s << " error: " << error << " iterarion: " << iter <<std::endl;
                if (N <=13)
                {
                    for (size_t i = 0; i < N; i++)
                    {
                        for (size_t j = 0; j < N; j++)
                        {
                            std::cout << Matr[i*N+j] << ' ';   
                        }
                        std::cout << std::endl;
                    }
                }
    saveMatrixToFile(std::ref(Matr), N , "Out_Matr.txt");
    Matr = nullptr;
    Matrnew = nullptr;  
    return 0;
}