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
    err = clGetPlatformIDs(2, platform.data(), NULL);
    checkErr(err, __LINE__);
    err = clGetDeviceIDs(platform[1], CL_DEVICE_TYPE_GPU, 0, NULL, &NumDevice);
    checkErr(err, __LINE__);
    printf("NumDevice:%d\n", NumDevice);

    device = (cl_device_id *)malloc(sizeof(cl_device_id) * NumDevice);
    //选择GPU设备
    err = clGetDeviceIDs(
            platform[1], CL_DEVICE_TYPE_GPU, NumDevice, device, NULL);
    checkErr(err, __LINE__);

    cl_context_properties properites[]
            = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform[1], 0};

    // 指定设备类型创建上下文
    // cl_context context = clCreateContextFromType(
    //         properites, CL_DEVICE_TYPE_ALL, NULL, NULL, &err);

    // 指定设备创建上下文
    cl_context context
            = clCreateContext(properites, NumDevice, device, NULL, NULL, &err);
    //     cl_context context
    //             = clCreateContext(NULL, NumDevice, device, NULL, NULL, &err);
    checkErr(err, __LINE__);
    NumDevice = 0;

    err = clGetContextInfo(
            context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint), &NumDevice, NULL);
    checkErr(err, __LINE__);
    printf("Number of Device in context:%d\n", NumDevice);

    cl_uint ReferenCount;
    clGetContextInfo(context, CL_CONTEXT_REFERENCE_COUNT, sizeof(cl_uint),
            &ReferenCount, NULL);
    printf("Initial Reference Count: %d\n ", ReferenCount);
    // Retain 接口增加引用计数
    clRetainContext(context);
    clGetContextInfo(context, CL_CONTEXT_REFERENCE_COUNT, sizeof(cl_uint),
            &ReferenCount, NULL);
    printf("Reference Count: %d\n ", ReferenCount);

    // Release 接口减少引用计数
    err = clReleaseContext(context);
    checkErr(err, __LINE__);
    err = clGetContextInfo(context, CL_CONTEXT_REFERENCE_COUNT, sizeof(cl_uint),
            &ReferenCount, NULL);
    checkErr(err, __LINE__);
    printf("Reference Count: %d\n ", ReferenCount);

    err = clReleaseContext(context);
    checkErr(err, __LINE__);

    // context 已被销毁，无法获取引用计数
    // err = clGetContextInfo(context, CL_CONTEXT_REFERENCE_COUNT, sizeof(cl_uint),
    //         &ReferenCount, NULL);
    // checkErr(err, __LINE__);
    // printf("Reference Count: %d\n ", ReferenCount);

    free(device);
    return 0;
}
