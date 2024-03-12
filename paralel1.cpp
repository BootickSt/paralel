#include  <iostream>
#include <vector>
#include <math.h>

const int size = 10000000;


int main(){

     #ifdef Float
        std::vector<float> array(size);
#else
        std::vector<double> array(size);
#endif

for (int i =0;i<size;i++){

        double corner = (2*M_PI*i) /size;
        array[i] = std::sin(corner);

}
double sum = 0.0;
for (int i = 0;i < size;i++){
        sum +=array[i];
}
std::cout << "result = " << sum << std::endl;

return 0;
}