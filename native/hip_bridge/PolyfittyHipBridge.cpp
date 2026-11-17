

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cwctype>
#include <filesystem>
#include <iterator>
#include <mutex>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define PF_EXPORT extern "C" __declspec(dllexport)
#define PF_CALL __cdecl

using HipGetDeviceCountFn = decltype(&hipGetDeviceCount);
using HipGetDevicePropertiesFn = decltype(&hipGetDeviceProperties);
using HipSetDeviceFn = decltype(&hipSetDevice);
using HipMallocFn = decltype(&hipMalloc);
using HipMemcpyFn = decltype(&hipMemcpy);
using HipFreeFn = decltype(&hipFree);
using HipDeviceSynchronizeFn = decltype(&hipDeviceSynchronize);
using HipModuleLoadDataFn = decltype(&hipModuleLoadData);
using HipModuleGetFunctionFn = decltype(&hipModuleGetFunction);
using HipModuleLaunchKernelFn = decltype(&hipModuleLaunchKernel);
using HipModuleUnloadFn = decltype(&hipModuleUnload);
using HipGetErrorStringFn = decltype(&hipGetErrorString);

using HiprtcCreateProgramFn = decltype(&hiprtcCreateProgram);
using HiprtcCompileProgramFn = decltype(&hiprtcCompileProgram);
using HiprtcGetProgramLogSizeFn = decltype(&hiprtcGetProgramLogSize);
using HiprtcGetProgramLogFn = decltype(&hiprtcGetProgramLog);
using HiprtcGetCodeSizeFn = decltype(&hiprtcGetCodeSize);
using HiprtcGetCodeFn = decltype(&hiprtcGetCode);
using HiprtcDestroyProgramFn = decltype(&hiprtcDestroyProgram);
using HiprtcGetErrorStringFn = decltype(&hiprtcGetErrorString);

static HMODULE g_hip_runtime = nullptr;
static HMODULE g_hiprtc = nullptr;
static std::mutex g_load_mutex;
static std::string g_last_error;

static HipGetDeviceCountFn p_hipGetDeviceCount = nullptr;
static HipGetDevicePropertiesFn p_hipGetDeviceProperties = nullptr;
static HipSetDeviceFn p_hipSetDevice = nullptr;
static HipMallocFn p_hipMalloc = nullptr;
static HipMemcpyFn p_hipMemcpy = nullptr;
static HipFreeFn p_hipFree = nullptr;
static HipDeviceSynchronizeFn p_hipDeviceSynchronize = nullptr;
static HipModuleLoadDataFn p_hipModuleLoadData = nullptr;
static HipModuleGetFunctionFn p_hipModuleGetFunction = nullptr;
static HipModuleLaunchKernelFn p_hipModuleLaunchKernel = nullptr;
static HipModuleUnloadFn p_hipModuleUnload = nullptr;
static HipGetErrorStringFn p_hipGetErrorString = nullptr;

static HiprtcCreateProgramFn p_hiprtcCreateProgram = nullptr;
static HiprtcCompileProgramFn p_hiprtcCompileProgram = nullptr;
static HiprtcGetProgramLogSizeFn p_hiprtcGetProgramLogSize = nullptr;
static HiprtcGetProgramLogFn p_hiprtcGetProgramLog = nullptr;
static HiprtcGetCodeSizeFn p_hiprtcGetCodeSize = nullptr;
static HiprtcGetCodeFn p_hiprtcGetCode = nullptr;
static HiprtcDestroyProgramFn p_hiprtcDestroyProgram = nullptr;
static HiprtcGetErrorStringFn p_hiprtcGetErrorString = nullptr;

static void set_error(char* out, int capacity, const std::string& message) {
    g_last_error = message;
    if (out && capacity > 0) {
        const size_t count = (std::min)(static_cast<size_t>(capacity - 1), message.size());
        std::memcpy(out, message.data(), count);
        out[count] = '\0';
    }
}

static std::vector<fs::path> search_directories() {
    std::vector<fs::path> dirs;
    auto add = [&dirs](const fs::path& path) {
        if (path.empty()) return;
        std::error_code ec;
        fs::path resolved = fs::weakly_canonical(path, ec);
        if (ec) resolved = path;
        for (const auto& current : dirs) {
            if (_wcsicmp(current.c_str(), resolved.c_str()) == 0) return;
        }
        dirs.push_back(resolved);
    };

    wchar_t bundle_dir[32768]{};
    const DWORD bundle_len = GetEnvironmentVariableW(
        L"POLYFITTY_HIP_BUNDLE_DIR", bundle_dir, static_cast<DWORD>(std::size(bundle_dir))
    );
    if (bundle_len > 0 && bundle_len < std::size(bundle_dir)) {
        add(fs::path(bundle_dir));
        add(fs::path(bundle_dir).parent_path());
    }

    wchar_t executable_path[32768]{};
    const DWORD executable_len = GetModuleFileNameW(
        nullptr, executable_path, static_cast<DWORD>(std::size(executable_path))
    );
    if (executable_len > 0 && executable_len < std::size(executable_path)) {
        const fs::path executable_dir = fs::path(executable_path).parent_path();
        add(executable_dir / L"amd_hip" / L"bin");
        add(executable_dir / L"amd_hip");
    }

    for (const wchar_t* key : {L"HIP_PATH", L"ROCM_PATH"}) {
        wchar_t value[32768]{};
        const DWORD length = GetEnvironmentVariableW(key, value, static_cast<DWORD>(std::size(value)));
        if (length > 0 && length < std::size(value)) {
            add(fs::path(value) / L"bin");
            add(fs::path(value));
        }
    }

    wchar_t program_files[32768]{};
    const DWORD pf_len = GetEnvironmentVariableW(L"ProgramFiles", program_files, static_cast<DWORD>(std::size(program_files)));
    if (pf_len > 0 && pf_len < std::size(program_files)) {
        const fs::path amd_root = fs::path(program_files) / L"AMD" / L"ROCm";
        std::error_code ec;
        if (fs::exists(amd_root, ec)) {
            std::vector<fs::path> versions;
            for (const auto& entry : fs::directory_iterator(amd_root, ec)) {
                if (entry.is_directory(ec)) versions.push_back(entry.path());
            }
            std::sort(versions.rbegin(), versions.rend());
            for (const auto& version : versions) add(version / L"bin");
        }
    }

    wchar_t system_dir[MAX_PATH]{};
    const UINT system_len = GetSystemDirectoryW(system_dir, MAX_PATH);
    if (system_len > 0 && system_len < MAX_PATH) add(system_dir);

    wchar_t path_value[32768]{};
    const DWORD path_len = GetEnvironmentVariableW(L"PATH", path_value, static_cast<DWORD>(std::size(path_value)));
    if (path_len > 0 && path_len < std::size(path_value)) {
        std::wstring all(path_value, path_len);
        size_t start = 0;
        while (start <= all.size()) {
            const size_t end = all.find(L';', start);
            const std::wstring part = all.substr(start, end == std::wstring::npos ? end : end - start);
            if (!part.empty()) add(part);
            if (end == std::wstring::npos) break;
            start = end + 1;
        }
    }
    return dirs;
}

static std::vector<fs::path> find_matching_dlls(const std::wstring& prefix, const std::vector<fs::path>& dirs) {
    std::vector<fs::path> matches;
    for (const auto& dir : dirs) {
        std::error_code ec;
        if (!fs::is_directory(dir, ec)) continue;
        for (const auto& entry : fs::directory_iterator(dir, ec)) {
            if (ec || !entry.is_regular_file(ec)) continue;
            const std::wstring filename = entry.path().filename().wstring();
            const std::wstring lower = [&]() {
                std::wstring value = filename;
                std::transform(value.begin(), value.end(), value.begin(), [](wchar_t ch) { return static_cast<wchar_t>(towlower(ch)); });
                return value;
            }();
            std::wstring prefix_lower = prefix;
            std::transform(prefix_lower.begin(), prefix_lower.end(), prefix_lower.begin(), [](wchar_t ch) { return static_cast<wchar_t>(towlower(ch)); });
            if (lower.size() >= prefix_lower.size() + 4 && lower.rfind(prefix_lower, 0) == 0 && lower.substr(lower.size() - 4) == L".dll") {
                matches.push_back(entry.path());
            }
        }
    }
    std::sort(matches.begin(), matches.end(), [](const fs::path& a, const fs::path& b) {
        return a.filename().wstring() > b.filename().wstring();
    });
    return matches;
}

static bool resolve_hip_runtime() {
    if (g_hip_runtime) return true;
    const auto dirs = search_directories();
    const auto candidates = find_matching_dlls(L"amdhip64", dirs);
    for (const auto& candidate : candidates) {
        HMODULE module = LoadLibraryW(candidate.c_str());
        if (!module) continue;
        auto get = [module](const char* name) { return GetProcAddress(module, name); };
        p_hipGetDeviceCount = reinterpret_cast<HipGetDeviceCountFn>(get("hipGetDeviceCount"));
        p_hipGetDeviceProperties = reinterpret_cast<HipGetDevicePropertiesFn>(get("hipGetDeviceProperties"));
        p_hipSetDevice = reinterpret_cast<HipSetDeviceFn>(get("hipSetDevice"));
        p_hipMalloc = reinterpret_cast<HipMallocFn>(get("hipMalloc"));
        p_hipMemcpy = reinterpret_cast<HipMemcpyFn>(get("hipMemcpy"));
        p_hipFree = reinterpret_cast<HipFreeFn>(get("hipFree"));
        p_hipDeviceSynchronize = reinterpret_cast<HipDeviceSynchronizeFn>(get("hipDeviceSynchronize"));
        p_hipModuleLoadData = reinterpret_cast<HipModuleLoadDataFn>(get("hipModuleLoadData"));
        p_hipModuleGetFunction = reinterpret_cast<HipModuleGetFunctionFn>(get("hipModuleGetFunction"));
        p_hipModuleLaunchKernel = reinterpret_cast<HipModuleLaunchKernelFn>(get("hipModuleLaunchKernel"));
        p_hipModuleUnload = reinterpret_cast<HipModuleUnloadFn>(get("hipModuleUnload"));
        p_hipGetErrorString = reinterpret_cast<HipGetErrorStringFn>(get("hipGetErrorString"));
        if (p_hipGetDeviceCount && p_hipGetDeviceProperties && p_hipSetDevice && p_hipMalloc &&
            p_hipMemcpy && p_hipFree && p_hipDeviceSynchronize && p_hipModuleLoadData &&
            p_hipModuleGetFunction && p_hipModuleLaunchKernel && p_hipModuleUnload) {
            g_hip_runtime = module;
            return true;
        }
        FreeLibrary(module);
    }
    return false;
}

static bool resolve_hiprtc() {
    if (g_hiprtc) return true;
    const auto dirs = search_directories();
    const auto candidates = find_matching_dlls(L"hiprtc", dirs);
    for (const auto& candidate : candidates) {
        HMODULE module = LoadLibraryW(candidate.c_str());
        if (!module) continue;
        auto get = [module](const char* name) { return GetProcAddress(module, name); };
        p_hiprtcCreateProgram = reinterpret_cast<HiprtcCreateProgramFn>(get("hiprtcCreateProgram"));
        p_hiprtcCompileProgram = reinterpret_cast<HiprtcCompileProgramFn>(get("hiprtcCompileProgram"));
        p_hiprtcGetProgramLogSize = reinterpret_cast<HiprtcGetProgramLogSizeFn>(get("hiprtcGetProgramLogSize"));
        p_hiprtcGetProgramLog = reinterpret_cast<HiprtcGetProgramLogFn>(get("hiprtcGetProgramLog"));
        p_hiprtcGetCodeSize = reinterpret_cast<HiprtcGetCodeSizeFn>(get("hiprtcGetCodeSize"));
        p_hiprtcGetCode = reinterpret_cast<HiprtcGetCodeFn>(get("hiprtcGetCode"));
        p_hiprtcDestroyProgram = reinterpret_cast<HiprtcDestroyProgramFn>(get("hiprtcDestroyProgram"));
        p_hiprtcGetErrorString = reinterpret_cast<HiprtcGetErrorStringFn>(get("hiprtcGetErrorString"));
        if (p_hiprtcCreateProgram && p_hiprtcCompileProgram && p_hiprtcGetProgramLogSize &&
            p_hiprtcGetProgramLog && p_hiprtcGetCodeSize && p_hiprtcGetCode && p_hiprtcDestroyProgram) {
            g_hiprtc = module;
            return true;
        }
        FreeLibrary(module);
    }
    return false;
}

static const char* hip_error(hipError_t result) {
    if (p_hipGetErrorString) {
        const char* text = p_hipGetErrorString(result);
        if (text) return text;
    }
    return "unknown HIP runtime error";
}

static bool ensure_runtime(char* error, int capacity) {
    std::lock_guard<std::mutex> guard(g_load_mutex);
    if (!resolve_hip_runtime()) {
        set_error(error, capacity, "AMD HIP runtime DLL (amdhip64*.dll) was not found or did not expose the required APIs. Install a compatible AMD graphics driver with HIP Runtime support.");
        return false;
    }
    return true;
}

static bool ensure_hiprtc(char* error, int capacity) {
    std::lock_guard<std::mutex> guard(g_load_mutex);
    if (!resolve_hip_runtime()) {
        set_error(error, capacity, "AMD HIP runtime DLL (amdhip64*.dll) was not found. Install a compatible AMD graphics driver.");
        return false;
    }
    if (!resolve_hiprtc()) {
        set_error(error, capacity, "HIPRTC DLL (hiprtc*.dll) was not found. Rebuild the portable EXE with the AMD HIP SDK runtime libraries included.");
        return false;
    }
    return true;
}

struct HipBackendContext {
    int device_index = 0;
    hipModule_t module{};
    hipModule_t qem_module{};
    hipFunction_t intersect_kernel{};
    hipFunction_t closest_kernel{};
    hipFunction_t qem_kernel{};
};

static int compile_backend(int device_index, const char* source, HipBackendContext** out_context, char* error, int capacity) {
    if (!out_context || !source) {
        set_error(error, capacity, "Invalid HIP backend initialization arguments.");
        return 1;
    }
    *out_context = nullptr;
    if (!ensure_hiprtc(error, capacity)) return 2;

    hipError_t runtime_result = p_hipSetDevice(device_index);
    if (runtime_result != hipSuccess) {
        set_error(error, capacity, std::string("hipSetDevice failed: ") + hip_error(runtime_result));
        return 3;
    }
    hipDeviceProp_t properties{};
    runtime_result = p_hipGetDeviceProperties(&properties, device_index);
    if (runtime_result != hipSuccess) {
        set_error(error, capacity, std::string("hipGetDeviceProperties failed: ") + hip_error(runtime_result));
        return 4;
    }
    if (properties.gcnArchName[0] == '\0') {
        set_error(error, capacity, "AMD GPU did not expose gcnArchName; HIPRTC cannot select a safe compilation target.");
        return 5;
    }

    hiprtcProgram program{};
    hiprtcResult rtc_result = p_hiprtcCreateProgram(
        &program, source, "polyfitty_gpu_kernels.hip", 0, nullptr, nullptr
    );
    if (rtc_result != HIPRTC_SUCCESS) {
        const char* detail = p_hiprtcGetErrorString ? p_hiprtcGetErrorString(rtc_result) : "unknown HIPRTC error";
        set_error(error, capacity, std::string("hiprtcCreateProgram failed: ") + (detail ? detail : "unknown"));
        return 6;
    }

    const std::string architecture_option = std::string("--gpu-architecture=") + properties.gcnArchName;
    const char* options[] = { architecture_option.c_str() };
    rtc_result = p_hiprtcCompileProgram(program, 1, options);
    if (rtc_result != HIPRTC_SUCCESS) {
        size_t log_size = 0;
        p_hiprtcGetProgramLogSize(program, &log_size);
        std::string log(log_size ? log_size : 1, '\0');
        if (log_size) p_hiprtcGetProgramLog(program, log.data());
        const char* detail = p_hiprtcGetErrorString ? p_hiprtcGetErrorString(rtc_result) : "unknown HIPRTC error";
        p_hiprtcDestroyProgram(&program);
        set_error(error, capacity, std::string("HIPRTC compile failed for ") + properties.gcnArchName + " (" + (detail ? detail : "unknown") + "): " + log);
        return 7;
    }

    size_t code_size = 0;
    rtc_result = p_hiprtcGetCodeSize(program, &code_size);
    if (rtc_result != HIPRTC_SUCCESS || code_size == 0) {
        p_hiprtcDestroyProgram(&program);
        set_error(error, capacity, "hiprtcGetCodeSize failed or returned an empty code object.");
        return 8;
    }
    std::vector<unsigned char> code(code_size);
    rtc_result = p_hiprtcGetCode(program, reinterpret_cast<char*>(code.data()));
    p_hiprtcDestroyProgram(&program);
    if (rtc_result != HIPRTC_SUCCESS) {
        set_error(error, capacity, "hiprtcGetCode failed while retrieving the AMD kernel code object.");
        return 9;
    }

    auto* context = new HipBackendContext();
    context->device_index = device_index;
    runtime_result = p_hipModuleLoadData(&context->module, code.data());
    if (runtime_result != hipSuccess) {
        delete context;
        set_error(error, capacity, std::string("hipModuleLoadData failed: ") + hip_error(runtime_result));
        return 10;
    }
    runtime_result = p_hipModuleGetFunction(&context->intersect_kernel, context->module, "intersect_rays");
    if (runtime_result == hipSuccess) {
        runtime_result = p_hipModuleGetFunction(&context->closest_kernel, context->module, "closest_points");
    }
    if (runtime_result != hipSuccess) {
        p_hipModuleUnload(context->module);
        delete context;
        set_error(error, capacity, std::string("HIP kernel symbol lookup failed: ") + hip_error(runtime_result));
        return 11;
    }
    *out_context = context;
    return 0;
}

static bool allocate_and_copy(const void* host, size_t bytes, void** device, char* error, int capacity) {
    *device = nullptr;
    if (bytes == 0) return true;
    hipError_t result = p_hipMalloc(device, bytes);
    if (result != hipSuccess) {
        set_error(error, capacity, std::string("hipMalloc failed: ") + hip_error(result));
        return false;
    }
    result = p_hipMemcpy(*device, host, bytes, hipMemcpyHostToDevice);
    if (result != hipSuccess) {
        p_hipFree(*device);
        *device = nullptr;
        set_error(error, capacity, std::string("hipMemcpy host-to-device failed: ") + hip_error(result));
        return false;
    }
    return true;
}

static bool allocate_output(size_t bytes, void** device, char* error, int capacity) {
    *device = nullptr;
    if (bytes == 0) return true;
    hipError_t result = p_hipMalloc(device, bytes);
    if (result != hipSuccess) {
        set_error(error, capacity, std::string("hipMalloc output failed: ") + hip_error(result));
        return false;
    }
    return true;
}

static void free_all(std::vector<void*>& allocations) {
    for (auto it = allocations.rbegin(); it != allocations.rend(); ++it) {
        if (*it) p_hipFree(*it);
    }
    allocations.clear();
}

static bool launch(HipBackendContext* context, hipFunction_t kernel, unsigned long long total,
                   void* arguments, size_t argument_size, char* error, int capacity) {
    if (total == 0) return true;
    const unsigned int block_size = 256;
    const unsigned int grid_size = static_cast<unsigned int>((total + block_size - 1) / block_size);
    size_t size = argument_size;
    void* config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, arguments,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
        HIP_LAUNCH_PARAM_END
    };
    hipError_t result = p_hipModuleLaunchKernel(
        kernel, grid_size, 1, 1, block_size, 1, 1, 0, nullptr, nullptr, config
    );
    if (result == hipSuccess) result = p_hipDeviceSynchronize();
    if (result != hipSuccess) {
        set_error(error, capacity, std::string("HIP kernel launch/synchronization failed: ") + hip_error(result));
        return false;
    }
    (void)context;
    return true;
}

PF_EXPORT int PF_CALL pfhip_device_count(int* count, char* error, int capacity) {
    if (!count) { set_error(error, capacity, "Device count output pointer is null."); return 1; }
    if (!ensure_runtime(error, capacity)) return 2;
    int local_count = 0;
    const hipError_t result = p_hipGetDeviceCount(&local_count);
    if (result != hipSuccess) {
        set_error(error, capacity, std::string("hipGetDeviceCount failed: ") + hip_error(result));
        return 3;
    }
    *count = local_count;
    return 0;
}

PF_EXPORT int PF_CALL pfhip_device_info(int index, char* name, int name_capacity,
                                        char* arch, int arch_capacity,
                                        unsigned long long* memory_bytes, int* integrated,
                                        char* error, int capacity) {
    if (!ensure_runtime(error, capacity)) return 1;
    hipDeviceProp_t properties{};
    const hipError_t result = p_hipGetDeviceProperties(&properties, index);
    if (result != hipSuccess) {
        set_error(error, capacity, std::string("hipGetDeviceProperties failed: ") + hip_error(result));
        return 2;
    }
    if (name && name_capacity > 0) {
        strncpy_s(name, static_cast<size_t>(name_capacity), properties.name, _TRUNCATE);
    }
    if (arch && arch_capacity > 0) {
        strncpy_s(arch, static_cast<size_t>(arch_capacity), properties.gcnArchName, _TRUNCATE);
    }
    if (memory_bytes) *memory_bytes = static_cast<unsigned long long>(properties.totalGlobalMem);
    if (integrated) *integrated = properties.integrated ? 1 : 0;
    return 0;
}

PF_EXPORT int PF_CALL pfhip_create_backend(int index, const char* source, void** handle, char* error, int capacity) {
    return compile_backend(index, source, reinterpret_cast<HipBackendContext**>(handle), error, capacity);
}

PF_EXPORT void PF_CALL pfhip_destroy_backend(void* handle) {
    auto* context = static_cast<HipBackendContext*>(handle);
    if (!context) return;
    if (p_hipModuleUnload && context->qem_module) p_hipModuleUnload(context->qem_module);
    if (p_hipModuleUnload && context->module) p_hipModuleUnload(context->module);
    delete context;
}

PF_EXPORT int PF_CALL pfhip_intersect_rays(
    void* handle, size_t sample_count, size_t candidate_count,
    const double* origins, const double* directions, const double* triangles,
    double epsilon, double ray_limit, double* out_t, double* out_u, double* out_v,
    char* error, int capacity) {
    auto* context = static_cast<HipBackendContext*>(handle);
    if (!context || !origins || !directions || !triangles || !out_t || !out_u || !out_v) {
        set_error(error, capacity, "Invalid ray-intersection arguments."); return 1;
    }
    if (candidate_count > 0xFFFFFFFFULL) { set_error(error, capacity, "Candidate count exceeds the HIP kernel's uint32 limit."); return 2; }
    if (!ensure_runtime(error, capacity)) return 3;
    const hipError_t selected = p_hipSetDevice(context->device_index);
    if (selected != hipSuccess) { set_error(error, capacity, std::string("hipSetDevice failed: ") + hip_error(selected)); return 4; }
    const size_t ray_count = sample_count * 2;
    const size_t triangle_count = sample_count * candidate_count;
    const size_t output_count = ray_count * candidate_count;
    std::vector<void*> allocations;
    void *d_origins = nullptr, *d_directions = nullptr, *d_triangles = nullptr;
    void *d_t = nullptr, *d_u = nullptr, *d_v = nullptr;
    auto cleanup = [&]() { free_all(allocations); };
    if (!allocate_and_copy(origins, ray_count * 3 * sizeof(double), &d_origins, error, capacity)) return 5;
    allocations.push_back(d_origins);
    if (!allocate_and_copy(directions, ray_count * 3 * sizeof(double), &d_directions, error, capacity)) { cleanup(); return 6; }
    allocations.push_back(d_directions);
    if (!allocate_and_copy(triangles, triangle_count * 9 * sizeof(double), &d_triangles, error, capacity)) { cleanup(); return 7; }
    allocations.push_back(d_triangles);
    if (!allocate_output(output_count * sizeof(double), &d_t, error, capacity)) { cleanup(); return 8; }
    allocations.push_back(d_t);
    if (!allocate_output(output_count * sizeof(double), &d_u, error, capacity)) { cleanup(); return 9; }
    allocations.push_back(d_u);
    if (!allocate_output(output_count * sizeof(double), &d_v, error, capacity)) { cleanup(); return 10; }
    allocations.push_back(d_v);

    struct Args { const double* origins; const double* directions; const double* triangles; unsigned int candidate_count; double epsilon; double ray_limit; double* out_t; double* out_u; double* out_v; };
    Args args{static_cast<const double*>(d_origins), static_cast<const double*>(d_directions),
              static_cast<const double*>(d_triangles), static_cast<unsigned int>(candidate_count),
              epsilon, ray_limit, static_cast<double*>(d_t), static_cast<double*>(d_u), static_cast<double*>(d_v)};
    if (!launch(context, context->intersect_kernel, static_cast<unsigned long long>(output_count), &args, sizeof(args), error, capacity)) { cleanup(); return 11; }
    hipError_t result = p_hipMemcpy(out_t, d_t, output_count * sizeof(double), hipMemcpyDeviceToHost);
    if (result == hipSuccess) result = p_hipMemcpy(out_u, d_u, output_count * sizeof(double), hipMemcpyDeviceToHost);
    if (result == hipSuccess) result = p_hipMemcpy(out_v, d_v, output_count * sizeof(double), hipMemcpyDeviceToHost);
    cleanup();
    if (result != hipSuccess) { set_error(error, capacity, std::string("HIP output copy failed: ") + hip_error(result)); return 12; }
    return 0;
}

PF_EXPORT int PF_CALL pfhip_closest_points(
    void* handle, size_t sample_count, size_t candidate_count,
    const double* points, const double* triangles,
    double* out_distance, double* out_u, double* out_v,
    char* error, int capacity) {
    auto* context = static_cast<HipBackendContext*>(handle);
    if (!context || !points || !triangles || !out_distance || !out_u || !out_v) {
        set_error(error, capacity, "Invalid closest-point arguments."); return 1;
    }
    if (candidate_count > 0xFFFFFFFFULL) { set_error(error, capacity, "Candidate count exceeds the HIP kernel's uint32 limit."); return 2; }
    if (!ensure_runtime(error, capacity)) return 3;
    const hipError_t selected = p_hipSetDevice(context->device_index);
    if (selected != hipSuccess) { set_error(error, capacity, std::string("hipSetDevice failed: ") + hip_error(selected)); return 4; }
    const size_t triangle_count = sample_count * candidate_count;
    const size_t output_count = triangle_count;
    std::vector<void*> allocations;
    void *d_points = nullptr, *d_triangles = nullptr, *d_distance = nullptr, *d_u = nullptr, *d_v = nullptr;
    auto cleanup = [&]() { free_all(allocations); };
    if (!allocate_and_copy(points, sample_count * 3 * sizeof(double), &d_points, error, capacity)) return 5;
    allocations.push_back(d_points);
    if (!allocate_and_copy(triangles, triangle_count * 9 * sizeof(double), &d_triangles, error, capacity)) { cleanup(); return 6; }
    allocations.push_back(d_triangles);
    if (!allocate_output(output_count * sizeof(double), &d_distance, error, capacity)) { cleanup(); return 7; }
    allocations.push_back(d_distance);
    if (!allocate_output(output_count * sizeof(double), &d_u, error, capacity)) { cleanup(); return 8; }
    allocations.push_back(d_u);
    if (!allocate_output(output_count * sizeof(double), &d_v, error, capacity)) { cleanup(); return 9; }
    allocations.push_back(d_v);

    struct Args { const double* points; const double* triangles; unsigned int candidate_count; double* out_distance_squared; double* out_u; double* out_v; };
    Args args{static_cast<const double*>(d_points), static_cast<const double*>(d_triangles),
              static_cast<unsigned int>(candidate_count), static_cast<double*>(d_distance),
              static_cast<double*>(d_u), static_cast<double*>(d_v)};
    if (!launch(context, context->closest_kernel, static_cast<unsigned long long>(output_count), &args, sizeof(args), error, capacity)) { cleanup(); return 10; }
    hipError_t result = p_hipMemcpy(out_distance, d_distance, output_count * sizeof(double), hipMemcpyDeviceToHost);
    if (result == hipSuccess) result = p_hipMemcpy(out_u, d_u, output_count * sizeof(double), hipMemcpyDeviceToHost);
    if (result == hipSuccess) result = p_hipMemcpy(out_v, d_v, output_count * sizeof(double), hipMemcpyDeviceToHost);
    cleanup();
    if (result != hipSuccess) { set_error(error, capacity, std::string("HIP output copy failed: ") + hip_error(result)); return 11; }
    return 0;
}

static bool compile_qem_candidate_kernel(
    HipBackendContext* context, const char* source, char* error, int capacity) {
    if (!context) {
        set_error(error, capacity, "QEM HIP backend context is null.");
        return false;
    }
    if (context->qem_kernel) return true;
    if (!source || !*source) {
        set_error(error, capacity, "QEM HIPRTC source was not supplied for lazy compilation.");
        return false;
    }
    if (!ensure_hiprtc(error, capacity)) return false;

    const hipError_t selected = p_hipSetDevice(context->device_index);
    if (selected != hipSuccess) {
        set_error(error, capacity, std::string("hipSetDevice failed before QEM compilation: ") + hip_error(selected));
        return false;
    }
    hipDeviceProp_t properties{};
    const hipError_t property_result = p_hipGetDeviceProperties(&properties, context->device_index);
    if (property_result != hipSuccess) {
        set_error(error, capacity, std::string("hipGetDeviceProperties failed before QEM compilation: ") + hip_error(property_result));
        return false;
    }
    if (properties.gcnArchName[0] == '\0') {
        set_error(error, capacity, "AMD GPU did not expose gcnArchName for optional QEM compilation.");
        return false;
    }

    hiprtcProgram program{};
    hiprtcResult rtc_result = p_hiprtcCreateProgram(
        &program, source, "polyfitty_qem_candidates.hip", 0, nullptr, nullptr
    );
    if (rtc_result != HIPRTC_SUCCESS) {
        const char* detail = p_hiprtcGetErrorString ? p_hiprtcGetErrorString(rtc_result) : "unknown HIPRTC error";
        set_error(error, capacity, std::string("QEM hiprtcCreateProgram failed: ") + (detail ? detail : "unknown"));
        return false;
    }

    const std::string architecture_option = std::string("--gpu-architecture=") + properties.gcnArchName;
    const char* options[] = { architecture_option.c_str() };
    rtc_result = p_hiprtcCompileProgram(program, 1, options);
    if (rtc_result != HIPRTC_SUCCESS) {
        size_t log_size = 0;
        p_hiprtcGetProgramLogSize(program, &log_size);
        std::string log(log_size ? log_size : 1, '\0');
        if (log_size) p_hiprtcGetProgramLog(program, log.data());
        const char* detail = p_hiprtcGetErrorString ? p_hiprtcGetErrorString(rtc_result) : "unknown HIPRTC error";
        p_hiprtcDestroyProgram(&program);
        set_error(error, capacity, std::string("Optional QEM HIPRTC compile failed for ") + properties.gcnArchName
            + " (" + (detail ? detail : "unknown") + "): " + log);
        return false;
    }

    size_t code_size = 0;
    rtc_result = p_hiprtcGetCodeSize(program, &code_size);
    if (rtc_result != HIPRTC_SUCCESS || code_size == 0) {
        p_hiprtcDestroyProgram(&program);
        set_error(error, capacity, "QEM hiprtcGetCodeSize failed or returned an empty code object.");
        return false;
    }
    std::vector<unsigned char> code(code_size);
    rtc_result = p_hiprtcGetCode(program, reinterpret_cast<char*>(code.data()));
    p_hiprtcDestroyProgram(&program);
    if (rtc_result != HIPRTC_SUCCESS) {
        set_error(error, capacity, "QEM hiprtcGetCode failed while retrieving the optional kernel code object.");
        return false;
    }

    hipModule_t module{};
    hipError_t runtime_result = p_hipModuleLoadData(&module, code.data());
    if (runtime_result != hipSuccess) {
        set_error(error, capacity, std::string("QEM hipModuleLoadData failed: ") + hip_error(runtime_result));
        return false;
    }
    hipFunction_t kernel{};
    runtime_result = p_hipModuleGetFunction(&kernel, module, "qem_candidate_placements");
    if (runtime_result != hipSuccess) {
        p_hipModuleUnload(module);
        set_error(error, capacity, std::string("Optional QEM HIP kernel symbol lookup failed: ") + hip_error(runtime_result));
        return false;
    }
    context->qem_module = module;
    context->qem_kernel = kernel;
    return true;
}

PF_EXPORT int PF_CALL pfhip_qem_candidate_batch(
    void* handle, const char* qem_source, size_t vertex_count, size_t edge_count,
    const double* vertices, const double* quadrics,
    const unsigned long long* edges, double* out_cost, double* out_position,
    char* error, int capacity) {
    auto* context = static_cast<HipBackendContext*>(handle);
    if (!context || !vertices || !quadrics || !edges || !out_cost || !out_position) {
        set_error(error, capacity, "Invalid QEM candidate arguments."); return 1;
    }
    if (edge_count > 0xFFFFFFFFULL) {
        set_error(error, capacity, "QEM edge count exceeds the HIP kernel's uint32 limit."); return 2;
    }
    if (!ensure_runtime(error, capacity)) return 3;
    const hipError_t selected = p_hipSetDevice(context->device_index);
    if (selected != hipSuccess) {
        set_error(error, capacity, std::string("hipSetDevice failed: ") + hip_error(selected)); return 4;
    }
    if (!compile_qem_candidate_kernel(context, qem_source, error, capacity)) return 5;
    std::vector<void*> allocations;
    void *d_vertices = nullptr, *d_quadrics = nullptr, *d_edges = nullptr;
    void *d_cost = nullptr, *d_position = nullptr;
    auto cleanup = [&]() { free_all(allocations); };
    if (!allocate_and_copy(vertices, vertex_count * 3 * sizeof(double), &d_vertices, error, capacity)) return 6;
    allocations.push_back(d_vertices);
    if (!allocate_and_copy(quadrics, vertex_count * 16 * sizeof(double), &d_quadrics, error, capacity)) { cleanup(); return 7; }
    allocations.push_back(d_quadrics);
    if (!allocate_and_copy(edges, edge_count * 2 * sizeof(unsigned long long), &d_edges, error, capacity)) { cleanup(); return 8; }
    allocations.push_back(d_edges);
    if (!allocate_output(edge_count * sizeof(double), &d_cost, error, capacity)) { cleanup(); return 9; }
    allocations.push_back(d_cost);
    if (!allocate_output(edge_count * 3 * sizeof(double), &d_position, error, capacity)) { cleanup(); return 10; }
    allocations.push_back(d_position);

    struct Args {
        const double* vertices;
        const double* quadrics;
        const unsigned long long* edges;
        unsigned int edge_count;
        double* out_cost;
        double* out_position;
    };
    Args args{
        static_cast<const double*>(d_vertices),
        static_cast<const double*>(d_quadrics),
        static_cast<const unsigned long long*>(d_edges),
        static_cast<unsigned int>(edge_count),
        static_cast<double*>(d_cost),
        static_cast<double*>(d_position),
    };
    if (!launch(context, context->qem_kernel, static_cast<unsigned long long>(edge_count), &args, sizeof(args), error, capacity)) {
        cleanup(); return 11;
    }
    hipError_t result = p_hipMemcpy(out_cost, d_cost, edge_count * sizeof(double), hipMemcpyDeviceToHost);
    if (result == hipSuccess) result = p_hipMemcpy(out_position, d_position, edge_count * 3 * sizeof(double), hipMemcpyDeviceToHost);
    cleanup();
    if (result != hipSuccess) {
        set_error(error, capacity, std::string("HIP QEM output copy failed: ") + hip_error(result)); return 12;
    }
    return 0;
}
