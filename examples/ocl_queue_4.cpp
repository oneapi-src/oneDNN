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
    printf("NumDevice:%d\n", NumDevice);

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

    // 打印队列属性（验证是否成功）
    cl_command_queue_properties queue_props;
    err = clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES,
            sizeof(cl_command_queue_properties), &queue_props, nullptr);
    checkErr(err, __LINE__);
    if (queue_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
        printf("Out-of-order queue enabled.\n");
    }

    err = clReleaseCommandQueue(queue);
    checkErr(err, __LINE__);

    err = clReleaseContext(context);
    checkErr(err, __LINE__);
    free(device);
    return 0;
}
