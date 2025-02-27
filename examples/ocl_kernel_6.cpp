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

    clBuildProgram(ocl_program, 0, nullptr, nullptr, nullptr, nullptr);
    checkErr(err, __LINE__);

    size_t num_kernels;
    err = clGetProgramInfo(ocl_program, CL_PROGRAM_NUM_KERNELS, sizeof(size_t),
            &num_kernels, nullptr);

    // 创建内核
    cl_kernel kernel_init = clCreateKernel(ocl_program, "init", &err);
    checkErr(err, __LINE__);
    cl_kernel kernel_add = clCreateKernel(ocl_program, "add", &err);
    checkErr(err, __LINE__);

    // 创建缓冲区（示例）
    cl_mem data_buffer, src0_buffer, src1_buffer, dst_buffer;
    const size_t element_count = 1024;
    const size_t buffer_size = element_count * sizeof(float);
    data_buffer = clCreateBuffer(
            context, CL_MEM_READ_WRITE, buffer_size, nullptr, &err);
    checkErr(err, __LINE__);
    src0_buffer = clCreateBuffer(
            context, CL_MEM_READ_ONLY, buffer_size, nullptr, &err);
    checkErr(err, __LINE__);
    src1_buffer = clCreateBuffer(
            context, CL_MEM_READ_ONLY, buffer_size, nullptr, &err);
    checkErr(err, __LINE__);
    dst_buffer = clCreateBuffer(
            context, CL_MEM_WRITE_ONLY, buffer_size, nullptr, &err);
    checkErr(err, __LINE__);

    // 初始化 src0 和 src1 的内容
    float *host_src0 = (float *)malloc(buffer_size);
    float *host_src1 = (float *)malloc(buffer_size);
    for (size_t i = 0; i < element_count; i++) {
        host_src0[i] = 1.0f;
        host_src1[i] = 2.0f;
    }
    // 将数据拷贝到设备
    err = clEnqueueWriteBuffer(queue, src0_buffer, CL_TRUE, 0, buffer_size,
            host_src0, 0, NULL, NULL);
    checkErr(err, __LINE__);
    err = clEnqueueWriteBuffer(queue, src1_buffer, CL_TRUE, 0, buffer_size,
            host_src1, 0, NULL, NULL);
    checkErr(err, __LINE__);

    // 执行 init 内核
    err = clSetKernelArg(kernel_init, 0, sizeof(cl_mem), &data_buffer);
    checkErr(err, __LINE__);
    size_t global_size = 1024;
    err = clEnqueueNDRangeKernel(queue, kernel_init, 1, nullptr, &global_size,
            nullptr, 0, nullptr, nullptr);
    checkErr(err, __LINE__);

    // 执行 add 内核（假设 src0 和 src1 已填充数据）
    err = clSetKernelArg(kernel_add, 0, sizeof(cl_mem), &src0_buffer);
    err |= clSetKernelArg(kernel_add, 1, sizeof(cl_mem), &src1_buffer);
    err |= clSetKernelArg(kernel_add, 2, sizeof(cl_mem), &dst_buffer);
    checkErr(err, __LINE__);
    err = clEnqueueNDRangeKernel(queue, kernel_add, 1, nullptr, &global_size,
            nullptr, 0, nullptr, NULL);
    checkErr(err, __LINE__);

    // 等待所有命令执行完成
    clFinish(queue);

    // 打印初始化后的 data_buffer 结果
    float *host_data = (float *)malloc(buffer_size);
    err = clEnqueueReadBuffer(queue, data_buffer, CL_TRUE, 0, buffer_size,
            host_data, 0, NULL, NULL);
    checkErr(err, __LINE__);

    printf("\n=== init kernel 结果 ===\n");
    for (int i = 0; i < 10; i++) {
        printf("data[%d] = %+.1f\n", i, host_data[i]);
    }

    // 打印加法后的 dst_buffer 结果
    float *host_dst = (float *)malloc(buffer_size);
    err = clEnqueueReadBuffer(queue, dst_buffer, CL_TRUE, 0, buffer_size,
            host_dst, 0, NULL, NULL);
    checkErr(err, __LINE__);

    printf("\n=== add kernel 结果 ==\n");
    for (int i = 0; i < 10; i++) {
        printf("dst[%d] = %.1f + %.1f = %.1f\n", i, host_src0[i], host_src1[i],
                host_dst[i]);
    }

    // 清理资源
    free(host_src0);
    free(host_src1);
    free(host_data);
    free(host_dst);
    clReleaseMemObject(data_buffer);
    clReleaseMemObject(src0_buffer);
    clReleaseMemObject(src1_buffer);
    clReleaseMemObject(dst_buffer);
    clReleaseKernel(kernel_init);
    clReleaseKernel(kernel_add);

    checkErr(err, __LINE__);

    clReleaseProgram(ocl_program);
    checkErr(err, __LINE__);

    err = clReleaseCommandQueue(queue);
    checkErr(err, __LINE__);

    err = clReleaseContext(context);
    checkErr(err, __LINE__);
    free(device);
    return 0;
}
