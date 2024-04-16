#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <memory>
#include <string>
#include <cstring>
#include <omp.h>

int n = 17000;
double tau = 0.0001;
double epsilon = 0.00001;

double cpuSecond()
{
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count() * 1e-9;
}


void algor(double* matr, double* vec, double* x, double* x1, double down, int num_threads)
{
    std::unique_ptr<double[]> upp(new double[n]);
    while(true)
    {
        double up = 0.0;
#pragma omp parallel for num_threads(num_threads)
        for (size_t i = 0; i < n; i++)
        {
            for (size_t j = 0; j < n; j++)
            {
                x1[i] += matr[i * n + j] * x[j];
            }
            x1[i] = x1[i] - vec[i];
            upp[i] = pow(x1[i], 2);
            x1[i] = x[i] - tau * x1[i];
#pragma omp atomic 
            up += upp[i];
        }
        if (sqrt(up)/sqrt(down) < epsilon)
        {
            break;
        }
        std::memcpy(x, x1, sizeof(double)*n);
    }
}

int main(int argc, char **argv){
std::unique_ptr<double[]> matr(new double[n*n]);
std::unique_ptr<double[]> vec(new double[n]);
std::unique_ptr<double[]> x(new double[n]);
std::unique_ptr<double[]> x1(new double[n]);
int num_threads = atoi(argv[1]);
for (size_t i = 0; i < n; i++)
{
    vec[i]= n + 1;
    x[i] = 0.0;
    x1[i] = 0.0;
   for (size_t j = 0; j < n; j++)
   {
    if (i == j)
    {
        matr[i * n + j] = 2.0;
    }
    else
    {
        matr[i*n+j]=1.0;
    }
    
   }
   
}
double down = pow(n + 1, 2) * n;

double time = cpuSecond();
   algor(matr.get(), vec.get(), x.get(), x1.get(), down, num_threads);
   
time = cpuSecond() - time;
for (size_t i = 0; i < 10; i++)
{
    std::cout<< 'x' << i << '=' << ' ' << x1[i] << ' ' << std::endl;
}
std:: cout << ' ' << time << "sec." << std::endl;
}