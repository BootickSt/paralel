#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

std::vector<std::vector<double>> read_matrix(const std::string& filename) {
    std::vector<std::vector<double>> matrix;
    std::ifstream file(filename);
    std::string line;

    if (!file) {
        std::cerr << "Не удалось открыть файл: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    while (getline(file, line)) {
        std::vector<double> row;
        std::istringstream iss(line);
        int num;

        while (iss >> num) {
            row.push_back(num);
        }
        matrix.push_back(row);
    }

    return matrix;
}

bool compare_matrices(const std::vector<std::vector<double>>& matrix1, const std::vector<std::vector<double>>& matrix2) {
    if (matrix1.size() != matrix2.size()) {
        return false;
    }

    for (size_t i = 0; i < matrix1.size(); i++) {
        if (matrix1[i].size() != matrix2[i].size()) {
            return false;
        }
        for (size_t j = 0; j < matrix1[i].size(); j++) {
            if (matrix1[i][j] != matrix2[i][j]) {
                return false;
            }
        }
    }

    return true;
}

int main(int argc, char** argv) {
    std::string filename1 = argv[1];
    std::string filename2 = argv[2];

    auto matrix1 = read_matrix(filename1);
    auto matrix2 = read_matrix(filename2);

    if (compare_matrices(matrix1, matrix2)) {
        std::cout << "Матрицы идентичны." << std::endl;
    } else {
        std::cout << "Матрицы не идентичны." << std::endl;
    }

    return 0;
}
