#include <stdio.h>
#include <string.h>
#include <sstream>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <dlfcn.h>
#define CUDAAPI
#define LOAD_FUNC(l, s) dlsym(l, s)
#define DL_CLOSE_FUNC(l) dlclose(l)

typedef enum nvmlReturn_enum
{
    NVML_SUCCESS = 0,                   //!< The operation was successful
    NVML_ERROR_UNINITIALIZED = 1,       //!< NVML was not first initialized with nvmlInit()
    NVML_ERROR_INVALID_ARGUMENT = 2,    //!< A supplied argument is invalid
    NVML_ERROR_NOT_SUPPORTED = 3,       //!< The requested operation is not available on target device
    NVML_ERROR_NO_PERMISSION = 4,       //!< The current user does not have permission for operation
    NVML_ERROR_ALREADY_INITIALIZED = 5, //!< Deprecated: Multiple initializations are now allowed through ref counting
    NVML_ERROR_NOT_FOUND = 6,           //!< A query to find an object was unsuccessful
    NVML_ERROR_INSUFFICIENT_SIZE = 7,   //!< An input argument is not large enough
    NVML_ERROR_INSUFFICIENT_POWER = 8,  //!< A device's external power cables are not properly attached
    NVML_ERROR_DRIVER_NOT_LOADED = 9,   //!< NVIDIA driver is not loaded
    NVML_ERROR_TIMEOUT = 10,            //!< User provided timeout passed
    NVML_ERROR_UNKNOWN = 999            //!< An internal driver error occurred
} nvmlReturn_t;

typedef void * nvmlDevice_t;
typedef struct nvmlMemory_st
{
    unsigned long long total;        //!< Total installed FB memory (in bytes)
    unsigned long long free;         //!< Unallocated FB memory (in bytes)
    unsigned long long used;         //!< Allocated FB memory (in bytes). Note that the driver/GPU always sets aside a small amount of memory for bookkeeping
} nvmlMemory_t;

typedef struct nvmlUtilization_st
{
    unsigned int gpu;                //!< Percent of time over the past second during which one or more kernels was executing on the GPU
    unsigned int memory;             //!< Percent of time over the past second during which global (device) memory was being read or written
} nvmlUtilization_t;


typedef nvmlReturn_t(CUDAAPI *NVMLINIT)(void);  // nvmlInit
typedef nvmlReturn_t(CUDAAPI *NVMLSHUTDOWN)(void);  // nvmlShutdown 
typedef nvmlReturn_t(CUDAAPI *NVMLDEVICEGETCOUNT)(unsigned int *deviceCount); // nvmlDeviceGetCount
typedef nvmlReturn_t(CUDAAPI *NVMLDEVICEGETHANDLEBYINDEX)(unsigned int index, nvmlDevice_t *device); // nvmlDeviceGetHandleByIndex
typedef nvmlReturn_t(CUDAAPI *NVMLDEVICEGETMEMORYINFO)(nvmlDevice_t device, nvmlMemory_t *memory); // nvmlDeviceGetMemoryInfo
typedef nvmlReturn_t(CUDAAPI *NVMLDEVICEGETUTILIZATIONRATES)(nvmlDevice_t device, nvmlUtilization_t *utilization); // nvmlDeviceGetUtilizationRates
typedef nvmlReturn_t(CUDAAPI *NVMLDEVICEGETTEMPERATURE)(nvmlDevice_t device, int sensorType, unsigned int *temp); // nvmlDeviceGetTemperature

#define GPU_MAX_SIZE    128


#define RETURN_SUCCESS     0
#define RETURN_ERROR_LOAD_LIB       (-1)
#define RETURN_ERROR_LOAD_FUNC      (-2)
#define RETURN_ERROR_LIB_FUNC       (-3)
#define RETURN_ERROR_NULL_POINTER   (-4)


#define CHECK_LOAD_NVML_FUNC(t, f, s) \
do { \
    (f) = (t)LOAD_FUNC(nvml_lib, s); \
    if (!(f)) { \
        printf("Failed loading %s from NVML library\n", s); \
        retCode = RETURN_ERROR_LOAD_FUNC; \
         goto gpu_fail;\
    } \
} while (0)

static int check_nvml_error(int err, const char *func)
{
    if (err != NVML_SUCCESS) {
        printf(" %s - failed with error code:%d\n", func, err);
        return 0;
    }
    return 1;
}

#define check_nvml_errors(f) \
do{ \
    if (!check_nvml_error(f, #f)) { \
        retCode = RETURN_ERROR_LIB_FUNC; \
        goto gpu_fail;\
    }\
}while(0)

void getdata() {

    int retCode = RETURN_SUCCESS;
    void* nvml_lib;
    NVMLINIT                    nvml_init;
    NVMLSHUTDOWN                nvml_shutdown;
    NVMLDEVICEGETCOUNT          nvml_device_get_count;
    NVMLDEVICEGETHANDLEBYINDEX  nvml_device_get_handle_by_index;
    NVMLDEVICEGETMEMORYINFO     nvml_device_get_memory_info;
    NVMLDEVICEGETUTILIZATIONRATES       nvml_device_get_utilization_rates;
    NVMLDEVICEGETTEMPERATURE    nvml_device_get_temperature;

    nvmlDevice_t device_handel;
     unsigned int utilization_value = 0;
    unsigned int utilization_sample = 0;
    int best_gpu = 0;
    unsigned int decoder_used = 100;

    // open the libnvidia-ml.so
    nvml_lib = NULL;
    nvml_lib = dlopen("libnvidia-ml.so", RTLD_NOW);
     if(nvml_lib == NULL){
        return;
    }
    unsigned int device_count = 0;
    nvmlMemory_t memory_info;
    nvmlUtilization_t gpu_utilization;
    int i = 0;

             CHECK_LOAD_NVML_FUNC(NVMLINIT, nvml_init, "nvmlInit");
    CHECK_LOAD_NVML_FUNC(NVMLSHUTDOWN, nvml_shutdown, "nvmlShutdown");

    CHECK_LOAD_NVML_FUNC(NVMLDEVICEGETCOUNT, nvml_device_get_count, "nvmlDeviceGetCount");
    CHECK_LOAD_NVML_FUNC(NVMLDEVICEGETHANDLEBYINDEX, nvml_device_get_handle_by_index, "nvmlDeviceGetHandleByIndex");
    CHECK_LOAD_NVML_FUNC(NVMLDEVICEGETMEMORYINFO, nvml_device_get_memory_info, "nvmlDeviceGetMemoryInfo");
    CHECK_LOAD_NVML_FUNC(NVMLDEVICEGETUTILIZATIONRATES, nvml_device_get_utilization_rates, "nvmlDeviceGetUtilizationRates");
    CHECK_LOAD_NVML_FUNC(NVMLDEVICEGETTEMPERATURE, nvml_device_get_temperature, "nvmlDeviceGetTemperature");
    check_nvml_errors(nvml_init());
    check_nvml_errors(nvml_device_get_count(&device_count));

    for(i = 0; i < device_count; i++){
        check_nvml_errors(nvml_device_get_handle_by_index(i, &device_handel));
        check_nvml_errors(nvml_device_get_memory_info(device_handel, &memory_info));
        check_nvml_errors(nvml_device_get_utilization_rates(device_handel, &gpu_utilization));
        printf("GPU:%d\t, Utilization:[gpu:%u, memory:%u], Memory:[used:%llu]\n ", gpu_utilization.gpu, gpu_utilization.memory,  memory_info.used);
    }

gpu_fail:
    nvml_shutdown();
    //关闭动态库
    dlclose(nvml_lib);


}

int main(int argc, char **argv)
{
       getdata();
    return 0;
}
