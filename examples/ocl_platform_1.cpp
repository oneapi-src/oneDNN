#include <malloc.h>
#include <stdio.h>
#include <string.h>
#include <CL/cl.h>
int main() {
    cl_platform_id *platform;
    cl_uint num_platform;
    cl_int err;
    err = clGetPlatformIDs(0, NULL, &num_platform);
    platform = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platform);
    err = clGetPlatformIDs(num_platform, platform, NULL);
    printf("\nnum_platformn: %d\n", num_platform);
    for (int i = 0; i < num_platform; i++) {
        size_t size;
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, 0, NULL, &size);
        char *PName = (char *)malloc(size);
        err = clGetPlatformInfo(
                platform[i], CL_PLATFORM_NAME, size, PName, NULL);
        printf("\nCL_PLATFORM_NAME: % s\n", PName);
        err = clGetPlatformInfo(
                platform[i], CL_PLATFORM_VENDOR, 0, NULL, &size);
        char *PVendor = (char *)malloc(size);
        err = clGetPlatformInfo(
                platform[i], CL_PLATFORM_VENDOR, size, PVendor, NULL);
        printf("CL_PLATFORM_VENDOR: % s\n", PVendor);
        err = clGetPlatformInfo(
                platform[i], CL_PLATFORM_VERSION, 0, NULL, &size);
        char *PVersion = (char *)malloc(size);
        err = clGetPlatformInfo(
                platform[i], CL_PLATFORM_VERSION, size, PVersion, NULL);
        printf("CL_PLATFORM_VERSION: % s\n", PVersion);
        err = clGetPlatformInfo(
                platform[i], CL_PLATFORM_PROFILE, 0, NULL, &size);
        char *PProfile = (char *)malloc(size);
        err = clGetPlatformInfo(
                platform[i], CL_PLATFORM_PROFILE, size, PProfile, NULL);
        printf("CL_PLATFORM_PROFILE: % s\n", PProfile);
        err = clGetPlatformInfo(
                platform[i], CL_PLATFORM_EXTENSIONS, 0, NULL, &size);
        char *PExten = (char *)malloc(size);
        err = clGetPlatformInfo(
                platform[i], CL_PLATFORM_EXTENSIONS, size, PExten, NULL);
        printf("CL_PLATFORM_EXTENSIONS: % s\n", PExten);
        free(PName);
        free(PVendor);
        free(PVersion);
        free(PProfile);
        free(PExten);
    }
}
