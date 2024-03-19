#include <iostream>
#include <fstream>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <mutex>
#include "utils.h"
#include <chrono>

using float16_t = uint16_t;
using float32_t = float;

// Comment/uncomment one of these to control output/index type.
#define OUTPUT_TY int64_t
// #define OUTPUT_TY int32_t

// Comment/uncomment one of these to control input/float type.
#define INPUT_TY float32_t
// #define INPUT_TY float16_t

constexpr uint32_t recordRuns = 100u;
constexpr int ARGMAX_LABEL = 7243;

template <typename DataT>
static inline void fillIndex(DataT* mat, uint32_t m, uint32_t n)
{
    for(int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; j++)
        {
        // Force only certain index to have a value, the rest is set to 0.
        mat[i * n + j] = j == ARGMAX_LABEL ? static_cast<DataT>(250.0) : static_cast<DataT>(0.0);
        }
    }
}

std::vector<char> readFileIntoVector(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);

    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return std::vector<char>();
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();
    return buffer;
}

void benchmark_module(size_t reductionSize) {
    int batchSize = 1;

    // Initialize input matrices
    // TODO: Add support for parallel dimension.
    std::vector<INPUT_TY> inputBuffer(batchSize * reductionSize);
    std::vector<size_t> outputBuffer(batchSize);

    fillIndex(inputBuffer.data(), batchSize, reductionSize);

    std::cout << "Initializing device data..." << std::endl;

    // Allocate and copy device memory
    INPUT_TY *d_input;
    OUTPUT_TY *d_output;
    size_t* d_reductionSize;

    const size_t bytesInput = inputBuffer.size() * sizeof(INPUT_TY);
    const size_t bytesOutput = outputBuffer.size() * sizeof(OUTPUT_TY);

    CHECK_HIP_ERROR(hipMalloc(&d_input, bytesInput));
    CHECK_HIP_ERROR(hipMalloc(&d_output, bytesOutput));
    CHECK_HIP_ERROR(hipMalloc(&d_reductionSize, sizeof(size_t)));

    CHECK_HIP_ERROR(hipMemcpy(d_input, inputBuffer.data(), bytesInput, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_reductionSize, &reductionSize, sizeof(size_t), hipMemcpyHostToDevice));

    hipModule_t module;
    hipFunction_t kernel;
    std::vector<char> hsacoVec = readFileIntoVector("argmax_ukernel.c.hsaco");
    if (hipModuleLoadDataEx(&module, hsacoVec.data(), 0, NULL, NULL) != hipSuccess) {
      std::cout<<"Failed to load module!\n";
      return;
    }
    if (hipModuleGetFunction(&kernel, module, "argmax_F32I64") != hipSuccess) {
      std::cout<<"Failed to get function!\n";
      return;
    }

    // Setting up grid dim, block size, and pointer.
    size_t block_dimx = 32;
    size_t block_dimy = 1;
    int gridX = batchSize;
    int gridY = 1;
    void* kernelParam[] = {d_input, d_output, d_reductionSize};
    auto size = sizeof(kernelParam);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kernelParam,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};

    std::cout << "Launching Argmax kernel..." << std::endl;
    hipEvent_t startEvent, stopEvent;
    CHECK_HIP_ERROR(hipEventCreate(&startEvent));
    CHECK_HIP_ERROR(hipEventCreate(&stopEvent));

    CHECK_HIP_ERROR(hipEventRecord(startEvent));

    // Launching Kernel Begin
    for (uint32_t i = 0; i < recordRuns; ++i) {
      assert (hipModuleLaunchKernel(
          kernel, gridX, gridY, 1, block_dimx, block_dimy, 1,
          0, nullptr, nullptr, config) == 0);
    }
    CHECK_HIP_ERROR(hipEventRecord(stopEvent));
    CHECK_HIP_ERROR(hipEventSynchronize(stopEvent));

    auto elapsedTimeMs = 0.0f;
    CHECK_HIP_ERROR(hipEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));
    CHECK_HIP_ERROR(hipEventDestroy(startEvent));
    CHECK_HIP_ERROR(hipEventDestroy(stopEvent));

    hipMemcpy(outputBuffer.data(), d_output, bytesOutput, hipMemcpyDeviceToHost);
    std::cout<<"Argmax result:"<<outputBuffer[0]<<"\n";
    assert(outputBuffer[0] == ARGMAX_LABEL && "Expected argmax to match label!");
    std::cout<<"argmax kernel successfully match label!\n";

    // Release device memoryv
    CHECK_HIP_ERROR(hipFree(d_input));
    CHECK_HIP_ERROR(hipFree(d_output));
    std::cout<< "Average time per run is:" << elapsedTimeMs/recordRuns <<" ms/iter\n";
    std::cout << "Finished!" << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " reductionSize" << std::endl;
    return 1;
  }
  size_t reductionSize = atoi(argv[1]);
  benchmark_module(reductionSize);
  return 0;
}
