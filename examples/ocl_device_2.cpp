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
    device = (cl_device_id *)malloc(sizeof(cl_device_id) * NumDevice);
    //选择GPU设备
    err = clGetDeviceIDs(
            platform[1], CL_DEVICE_TYPE_GPU, NumDevice, device, NULL);
    checkErr(err, __LINE__);

    printf("NumDevice:%d\n", NumDevice);
    for (int i = 0; i < 1; i++) {
        //查询设备名称
        char buffer[100];
        err = clGetDeviceInfo(device[i], CL_DEVICE_NAME, 100, buffer, NULL);
        checkErr(err, __LINE__);
        printf("Device Name:%s\n", buffer);
        //查询设备并行计算单元最大数目:  intel(EU), cuda(SM)
        cl_uint UnitNum;
        err = clGetDeviceInfo(device[i], CL_DEVICE_MAX_COMPUTE_UNITS,
                sizeof(cl_uint), &UnitNum, NULL);
        checkErr(err, __LINE__);
        printf("Compute Units Number: %d\n", UnitNum);

        //查询全局和局部工作项最大维度
        size_t max_workitem_ndims;
        clGetDeviceInfo(*device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                sizeof(size_t), &max_workitem_ndims, NULL);
        checkErr(err, __LINE__);
        printf("Max_workitex_ndims: %lu\n", max_workitem_ndims);

        //查询单个工作组最大workitem数目
        size_t max_workgroup_size;
        clGetDeviceInfo(*device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
                &max_workgroup_size, NULL);
        checkErr(err, __LINE__);
        printf("Max workgroup size: %lu\n", max_workgroup_size);
        //查询设备核心频率
        cl_uint frequency;
        err = clGetDeviceInfo(device[i], CL_DEVICE_MAX_CLOCK_FREQUENCY,
                sizeof(cl_uint), &frequency, NULL);
        checkErr(err, __LINE__);
        printf("Device Frequency: %d(MHz)\n", frequency);
        //查询设备全局内存大小
        cl_ulong GlobalSize;
        err = clGetDeviceInfo(device[i], CL_DEVICE_GLOBAL_MEM_SIZE,
                sizeof(cl_ulong), &GlobalSize, NULL);
        checkErr(err, __LINE__);
        printf("Device Global Size: %0.0f(MB)\n",
                (float)GlobalSize / 1024 / 1024);

        //查询设备shared local memory内存大小
        cl_ulong SlmSize;
        err = clGetDeviceInfo(device[i], CL_DEVICE_LOCAL_MEM_SIZE,
                sizeof(cl_ulong), &SlmSize, NULL);
        checkErr(err, __LINE__);
        printf("Device Shared local memory Size: %0.0f(KB)\n",
                (float)SlmSize / 1024);
        //查询设备全局内存缓存行
        cl_uint GlobalCacheLine;
        err = clGetDeviceInfo(device[i], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
                sizeof(cl_uint), &GlobalCacheLine, NULL);
        checkErr(err, __LINE__);
        printf("Device Global CacheLine: %d(Byte)\n", GlobalCacheLine);

#ifndef CL_DEVICE_IP_VERSION_INTEL
#define CL_DEVICE_IP_VERSION_INTEL 0x4250
#endif

        struct HWIPVersion {
            union {
                uint32_t raw;
                struct {
                    uint32_t revision : 6;
                    uint32_t reserved : 8;
                    uint32_t release : 8;
                    uint32_t architecture : 10;
                };
            };
        } version;

        cl_uint ipVersion = 0;
        err = clGetDeviceInfo(device[i], CL_DEVICE_IP_VERSION_INTEL,
                sizeof(ipVersion), &ipVersion, NULL);
        checkErr(err, __LINE__);
        printf("Device IP Version: %d\n", ipVersion);
        version.raw = ipVersion;

        //DG2
        printf("version.architecture: %d\n", version.architecture);
        printf("version.release: %d\n", version.release);

        //查询设备支持的OpenCL版本
        char DeviceVersion[100];
        err = clGetDeviceInfo(
                device[i], CL_DEVICE_VERSION, 100, DeviceVersion, NULL);
        checkErr(err, __LINE__);
        printf("Device Version:%s\n", DeviceVersion);
        //查询设备拓展名
        char *DeviceExtensions;
        size_t ExtenNum;
        err = clGetDeviceInfo(
                device[i], CL_DEVICE_EXTENSIONS, 0, NULL, &ExtenNum);
        checkErr(err, __LINE__);
        DeviceExtensions = (char *)malloc(ExtenNum);
        err = clGetDeviceInfo(device[i], CL_DEVICE_EXTENSIONS, ExtenNum,
                DeviceExtensions, NULL);
        checkErr(err, __LINE__);
        printf("Device Extensions:%s\n", DeviceExtensions);
        free(DeviceExtensions);
    }
    free(device);
    return 0;
}
