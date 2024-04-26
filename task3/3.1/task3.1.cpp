#include <iostream>
#include <vector>
#include <thread>
#include <memory>
#include <cstring>
#include <chrono>

double cpuSecond()
{
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count() * 1e-9;
}

void multiplyElements(double* arr1, double* arr2, double* result, size_t lb, size_t ub, size_t id, size_t m, size_t n) {
    
    for (size_t i = lb; i <= ub; ++i) {
         double sum = 0.0;
        for (size_t j = 0; j < n; j++)
        {
            sum += arr1[i * n + j] * arr2[j];
        }
        result[i] = sum;
    }
    
}

void multiplyElements_sumple(double* arr1, double* arr2, double* result, size_t m, size_t n) {
    
    for (size_t i = 0; i <= m; ++i) {
         double sum = 0.0;
        for (size_t j = 0; j < n; j++)
        {
            sum += arr1[i * n + j] * arr2[j];
        }
        result[i] = sum;
    }
    
}

int main(int argc, char** argv) {
    int n = 10000, m = 10000;
    double time_s, time_p;
    int nt = 2;
    nt = atoi(argv[1]);
    std::unique_ptr<double[]> arr1(new double[m*n]);
    std::unique_ptr<double[]> arr2(new double[n]);
    std::unique_ptr<double[]> result(new double[m]);

    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            arr1[i * n + j] = j;
        }
    }
    for (size_t i = 0; i < n; i++)
    {
       arr2[i] = i;
    }
    
    
    std::vector<std::jthread> threads;
    int items_per_thread = m / nt;
    time_p = cpuSecond();
    for (size_t i = 0; i < nt; ++i) {
        int lb = i * items_per_thread;
        int ub = (i == nt - 1) ? (m - 1) : (lb + items_per_thread - 1);
        threads.emplace_back(multiplyElements, arr1.get() , arr2.get() , result.get() , lb, ub, i, m, n);
    }

    for (auto& thread : threads) {
        thread.join();
    }
    time_p = cpuSecond() - time_p;
    time_s = cpuSecond();
    multiplyElements_sumple(arr1.get(), arr2.get(), result.get(), m, n );
    time_s = cpuSecond() - time_s;
    std::cout<< "paralel time = " << time_p << std::endl;
    std::cout<< "single time = " << time_s << std::endl;
    std::cout<< "speedup = " << time_s/time_p<< std::endl;
    return 0;
}
