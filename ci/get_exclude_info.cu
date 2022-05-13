#include <iostream>

int main() {
    int count;
    cudaGetDeviceCount(&count);
    if (count == 0) {
        throw std::runtime_error("No GPU Found");
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::string output{};
    if (prop.major < 8) {
        output = "*SM80*";
    }
    if (prop.major < 7) {
        output += ":*SM70*";
    }
    std::cout << output;
    return 0;
}
