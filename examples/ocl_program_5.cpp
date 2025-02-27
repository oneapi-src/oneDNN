#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <CL/cl.h>
void checkErr(cl_int err, int num) {
    if (CL_SUCCESS != err) {
        printf("OpenCL error(%d) at %d\n", err, num - 1);
        exit(1);
    }
}

int main(int argc, char **argv) {
    cl_device_id *device;
    std::vector<cl_platform_id> platform(2);
    cl_int err;
    cl_uint NumDevice;
    //选择第一个平台
    err = clGetPlatformIDs(2, platform.data(), nullptr);
    checkErr(err, __LINE__);
    err = clGetDeviceIDs(
            platform[1], CL_DEVICE_TYPE_GPU, 0, nullptr, &NumDevice);
    checkErr(err, __LINE__);

    device = (cl_device_id *)malloc(sizeof(cl_device_id) * NumDevice);
    //选择GPU设备
    err = clGetDeviceIDs(
            platform[1], CL_DEVICE_TYPE_GPU, NumDevice, device, nullptr);
    checkErr(err, __LINE__);

    cl_context_properties properites[]
            = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform[1], 0};

    // 指定设备类型创建上下文
    // cl_context context = clCreateContextFromType(
    //         properites, CL_DEVICE_TYPE_ALL, nullptr, nullptr, &err);

    // 指定设备创建上下文
    cl_context context = clCreateContext(
            properites, NumDevice, device, nullptr, nullptr, &err);

    // OpenCL 2.0 API
#ifdef CL_VERSION_2_0
    // cl_queue_properties props[] = {CL_QUEUE_PROPERTIES,
    //         CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(
            context, *device, nullptr, &err);
    checkErr(err, __LINE__);
#else
    // OpenCL 1.x API
    cl_command_queue_properties prop = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    cl_command_queue queue = clCreateCommandQueue(context, *device, prop, &err);
    checkErr(err, __LINE__);
#endif

    // kernel code
    const char *ocl_code
            = "__kernel void init(__global float *data) {"
              "    int id = get_global_id(0);"
              "    data[id] = (id % 2) ? -id : id;"
              "}"
              "__kernel void add(__global float *src0,__global float "
              "*src1,__global float *dst) {"
              "    int id = get_global_id(0);"
              "    dst[id] = src0[id] + src1[id];"
              "}";
    const char *sources[] = {ocl_code};

    // create program
    cl_program ocl_program
            = clCreateProgramWithSource(context, 1, sources, nullptr, &err);
    checkErr(err, __LINE__);
    // build and link program
    clBuildProgram(ocl_program, 0, nullptr, nullptr, nullptr, nullptr);
    checkErr(err, __LINE__);

    size_t num_kernels;
    err = clGetProgramInfo(ocl_program, CL_PROGRAM_NUM_KERNELS, sizeof(size_t),
            &num_kernels, nullptr);
    checkErr(err, __LINE__);
    printf("kernel numbers:%d\n", num_kernels);

    clReleaseProgram(ocl_program);
    checkErr(err, __LINE__);

    err = clReleaseCommandQueue(queue);
    checkErr(err, __LINE__);

    err = clReleaseContext(context);
    checkErr(err, __LINE__);
    free(device);
    return 0;
}
