#include "arena/profiler.hpp"
#include "arena/utils.hpp"
#include <spdlog/spdlog.h>
#include <cstdlib>
#include <unordered_map>
#include <string>

namespace arena {

// Note: Profiler (CUPTI) will not run if nsys or ncu used (They will reserve CUPTI context before the arena app)

struct KernelActivityData {
    uint64_t duration_ns = 0;
    int registers_per_thread = 0;
    int shared_memory = 0;
};

static std::unordered_map<std::string, KernelActivityData> s_activity_results;

static void CUPTIAPI activity_buffer_requested(
    uint8_t** buffer, size_t* size, size_t* maxNumRecords)
{
    *size = 16 * 1024;
    *buffer = (uint8_t*)malloc(*size);
    *maxNumRecords = 0;
}

static void CUPTIAPI activity_buffer_completed(
    CUcontext ctx, uint32_t streamId,
    uint8_t* buffer, size_t size, size_t validSize)
{
    CUpti_Activity* record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            auto* kernel = (CUpti_ActivityKernel4*)record;
            auto& data = s_activity_results[kernel->name];
            data.duration_ns = kernel->end - kernel->start;
            data.registers_per_thread = kernel->registersPerThread;
            data.shared_memory = kernel->staticSharedMemory + kernel->dynamicSharedMemory;
        }
    }
    free(buffer);
}


Profiler::Profiler() {
    init_activity();
}

Profiler::~Profiler() {
    cleanup_range_profiler();
    cleanup_activity();
}

void Profiler::init_activity() {
    if (activity_initialized_) return;

    auto result = cuptiActivityRegisterCallbacks(
        activity_buffer_requested, activity_buffer_completed);
    if (result != CUPTI_SUCCESS) {
        spdlog::get("profiler")->warn("CUPTI Activity registration failed");
        return;
    }

    spdlog::get("profiler")->info("CUPTI Activity API initialized");
    activity_initialized_ = true;
}

void Profiler::cleanup_activity() {
    if (!activity_initialized_) return;
    cuptiActivityFlushAll(0);
    activity_initialized_ = false;
}

void Profiler::init_range_profiler() {
    if (range_profiler_initialized_) return;
    auto log = spdlog::get("profiler");

    // chip name
    CUpti_Device_GetChipName_Params chipParams = {
        CUpti_Device_GetChipName_Params_STRUCT_SIZE};
    chipParams.deviceIndex = 0;
    check_cupti(cuptiDeviceGetChipName(&chipParams), "cuptiDeviceGetChipName");
    chip_name_ = chipParams.pChipName;
    log->info("Range Profiler chip: {}", chip_name_);

    // device profiler (must be initialized before counter availability query)
    CUpti_Profiler_Initialize_Params profInitParams = {
        CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    check_cupti(cuptiProfilerInitialize(&profInitParams),
        "cuptiProfilerInitialize");

    // counter availability
    CUcontext cuContext;
    cuCtxGetCurrent(&cuContext);

    CUpti_Profiler_GetCounterAvailability_Params availParams = {
        CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
    availParams.ctx = cuContext;
    check_cupti(cuptiProfilerGetCounterAvailability(&availParams),
        "cuptiProfilerGetCounterAvailability (size)");
    counter_availability_image_.resize(availParams.counterAvailabilityImageSize);
    availParams.pCounterAvailabilityImage = counter_availability_image_.data();
    check_cupti(cuptiProfilerGetCounterAvailability(&availParams),
        "cuptiProfilerGetCounterAvailability (data)");

    // Set chip-specific CUPTI metric names
    bool is_blackwell = chip_name_.find("GB") == 0;
    chip_metrics_.dram_read  = is_blackwell ? "dram__bytes_op_read.sum"  : "dram__bytes_read.sum";
    chip_metrics_.dram_write = is_blackwell ? "dram__bytes_op_write.sum" : "dram__bytes_write.sum";
    chip_metrics_.occupancy  = "sm__warps_active.avg.pct_of_peak_sustained_active";
    chip_metrics_.ipc        = "smsp__inst_executed.avg.per_cycle_active";

    range_profiler_initialized_ = true;
    log->info("Range Profiler initialized");
}

void Profiler::cleanup_range_profiler() {
    if (!range_profiler_initialized_) return;
    destroy_host_object();
    range_profiler_initialized_ = false;
}

CUpti_Profiler_Host_Object* Profiler::create_host_object() {
    CUpti_Profiler_Host_Initialize_Params hostInitParams = {
        CUpti_Profiler_Host_Initialize_Params_STRUCT_SIZE};
    hostInitParams.profilerType = CUPTI_PROFILER_TYPE_RANGE_PROFILER;
    hostInitParams.pChipName = chip_name_.c_str();
    hostInitParams.pCounterAvailabilityImage = counter_availability_image_.data();
    check_cupti(cuptiProfilerHostInitialize(&hostInitParams),
        "cuptiProfilerHostInitialize");
    host_object_ = hostInitParams.pHostObject;
    return host_object_;
}

void Profiler::destroy_host_object() {
    if (host_object_) {
        CUpti_Profiler_Host_Deinitialize_Params deinitParams = {
            CUpti_Profiler_Host_Deinitialize_Params_STRUCT_SIZE};
        deinitParams.pHostObject = host_object_;
        cuptiProfilerHostDeinitialize(&deinitParams);
        host_object_ = nullptr;
    }
}


Profiler::ProfileResult Profiler::collect_activity(KernelLaunchFn launch_fn) {
    ProfileResult result;

    s_activity_results.clear();
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);

    launch_fn();
    cuCtxSynchronize();

    cuptiActivityFlushAll(0);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);

    if (!s_activity_results.empty()) {
        auto log = spdlog::get("profiler");

        // sum durations across all kernels (handles multi-kernel ops like CUB or e.g. two-stage kernels)
        uint64_t total_ns = 0;
        for (const auto& [name, data] : s_activity_results) {
            total_ns += data.duration_ns;
            result.sub_kernels.push_back({name, data.duration_ns / 1e6f,
                data.registers_per_thread, data.shared_memory});
            log->debug("  GPU kernel: {} | {:.3f} ms | {} regs | {} B shmem",
                name, data.duration_ns / 1e6f, data.registers_per_thread, data.shared_memory);
        }
        result.kernel_time_ms = total_ns / 1e6f;

        // take metadata from the longest-running kernel (the "main" one) TODO: Will need to change that
        auto longest = std::max_element(s_activity_results.begin(), s_activity_results.end(),
            [](const auto& a, const auto& b) { return a.second.duration_ns < b.second.duration_ns; });
        result.registers_per_thread = longest->second.registers_per_thread;
        result.shared_memory_per_block = longest->second.shared_memory;

        if (s_activity_results.size() > 1) {
            log->debug("  GPU total: {:.3f} ms ({} kernels)", result.kernel_time_ms, s_activity_results.size());
        }
    }

    return result;
}

std::map<std::string, double> Profiler::collect_counters(
    KernelLaunchFn launch_fn, KernelLaunchFn reset_fn)
{
    auto log = spdlog::get("profiler");
    std::map<std::string, double> result;

    init_range_profiler();

    // CUPTI metric names for this chip (set in init_range_profiler)
    std::vector<std::string> cupti_names = {
        chip_metrics_.dram_read, chip_metrics_.dram_write,
        chip_metrics_.occupancy, chip_metrics_.ipc
    };
    // Corresponding logical keys returned to caller
    std::vector<const char*> logical_keys = {
        metric::DRAM_READ, metric::DRAM_WRITE,
        metric::OCCUPANCY, metric::IPC
    };

    // create a fresh host object per call (ConfigAddMetrics accumulates state)
    auto* hostObj = create_host_object();

    std::vector<const char*> metric_cstrs;
    for (const auto& m : cupti_names) metric_cstrs.push_back(m.c_str());

    CUcontext cuContext;
    cuCtxGetCurrent(&cuContext);

    // enable range profiler on context
    CUpti_RangeProfiler_Enable_Params enableParams = {
        CUpti_RangeProfiler_Enable_Params_STRUCT_SIZE};
    enableParams.ctx = cuContext;
    check_cupti(cuptiRangeProfilerEnable(&enableParams), "cuptiRangeProfilerEnable");
    auto* rpObject = enableParams.pRangeProfilerObject;

    // configure metrics
    CUpti_Profiler_Host_ConfigAddMetrics_Params addMetricsParams = {
        CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
    addMetricsParams.pHostObject = hostObj;
    addMetricsParams.ppMetricNames = metric_cstrs.data();
    addMetricsParams.numMetrics = metric_cstrs.size();
    check_cupti(cuptiProfilerHostConfigAddMetrics(&addMetricsParams),
        "cuptiProfilerHostConfigAddMetrics");

    // config image
    CUpti_Profiler_Host_GetConfigImageSize_Params configSizeParams = {
        CUpti_Profiler_Host_GetConfigImageSize_Params_STRUCT_SIZE};
    configSizeParams.pHostObject = hostObj;
    check_cupti(cuptiProfilerHostGetConfigImageSize(&configSizeParams),
        "cuptiProfilerHostGetConfigImageSize");

    std::vector<uint8_t> configImage(configSizeParams.configImageSize);

    CUpti_Profiler_Host_GetConfigImage_Params configParams = {
        CUpti_Profiler_Host_GetConfigImage_Params_STRUCT_SIZE};
    configParams.pHostObject = hostObj;
    configParams.pConfigImage = configImage.data();
    configParams.configImageSize = configImage.size();
    check_cupti(cuptiProfilerHostGetConfigImage(&configParams),
        "cuptiProfilerHostGetConfigImage");

    // number of passes
    CUpti_Profiler_Host_GetNumOfPasses_Params numPassesParams = {
        CUpti_Profiler_Host_GetNumOfPasses_Params_STRUCT_SIZE};
    numPassesParams.pConfigImage = configImage.data();
    numPassesParams.configImageSize = configImage.size();
    check_cupti(cuptiProfilerHostGetNumOfPasses(&numPassesParams),
        "cuptiProfilerHostGetNumOfPasses");
    log->debug("Range Profiler: {} passes for {} metrics",
        numPassesParams.numOfPasses, cupti_names.size());

    // counter data image
    CUpti_RangeProfiler_GetCounterDataSize_Params counterDataSizeParams = {
        CUpti_RangeProfiler_GetCounterDataSize_Params_STRUCT_SIZE};
    counterDataSizeParams.pRangeProfilerObject = rpObject;
    counterDataSizeParams.pMetricNames = metric_cstrs.data();
    counterDataSizeParams.numMetrics = metric_cstrs.size();
    counterDataSizeParams.maxNumOfRanges = 1;
    counterDataSizeParams.maxNumRangeTreeNodes = 1;
    check_cupti(cuptiRangeProfilerGetCounterDataSize(&counterDataSizeParams),
        "cuptiRangeProfilerGetCounterDataSize");

    std::vector<uint8_t> counterDataImage(counterDataSizeParams.counterDataSize, 0);

    CUpti_RangeProfiler_CounterDataImage_Initialize_Params initCDParams = {
        CUpti_RangeProfiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
    initCDParams.pRangeProfilerObject = rpObject;
    initCDParams.pCounterData = counterDataImage.data();
    initCDParams.counterDataSize = counterDataImage.size();
    check_cupti(cuptiRangeProfilerCounterDataImageInitialize(&initCDParams),
        "cuptiRangeProfilerCounterDataImageInitialize");

    // set config
    CUpti_RangeProfiler_SetConfig_Params setConfigParams = {
        CUpti_RangeProfiler_SetConfig_Params_STRUCT_SIZE};
    setConfigParams.pRangeProfilerObject = rpObject;
    setConfigParams.pConfig = configImage.data();
    setConfigParams.configSize = configImage.size();
    setConfigParams.pCounterDataImage = counterDataImage.data();
    setConfigParams.counterDataImageSize = counterDataImage.size();
    setConfigParams.maxRangesPerPass = 1;
    setConfigParams.numNestingLevels = 1;
    setConfigParams.minNestingLevel = 1;
    setConfigParams.passIndex = 0;
    setConfigParams.targetNestingLevel = 1;
    setConfigParams.range = CUPTI_AutoRange;
    setConfigParams.replayMode = CUPTI_UserReplay;
    check_cupti(cuptiRangeProfilerSetConfig(&setConfigParams),
        "cuptiRangeProfilerSetConfig");

    // multi-pass replay
    bool allPassesDone = false;
    int passIndex = 0;
    while (!allPassesDone) {
        CUpti_RangeProfiler_Start_Params startParams = {
            CUpti_RangeProfiler_Start_Params_STRUCT_SIZE};
        startParams.pRangeProfilerObject = rpObject;
        check_cupti(cuptiRangeProfilerStart(&startParams), "cuptiRangeProfilerStart");

        if (reset_fn) reset_fn();
        launch_fn();
        cuCtxSynchronize();

        CUpti_RangeProfiler_Stop_Params stopParams = {
            CUpti_RangeProfiler_Stop_Params_STRUCT_SIZE};
        stopParams.pRangeProfilerObject = rpObject;
        check_cupti(cuptiRangeProfilerStop(&stopParams), "cuptiRangeProfilerStop");

        allPassesDone = stopParams.isAllPassSubmitted;
        passIndex++;
        log->debug("Range Profiler: pass {}/{}", passIndex, numPassesParams.numOfPasses);
    }

    // decode
    CUpti_RangeProfiler_DecodeData_Params decodeParams = {
        CUpti_RangeProfiler_DecodeData_Params_STRUCT_SIZE};
    decodeParams.pRangeProfilerObject = rpObject;
    check_cupti(cuptiRangeProfilerDecodeData(&decodeParams),
        "cuptiRangeProfilerDecodeData");

    // evaluate
    std::vector<double> metricValues(cupti_names.size());
    CUpti_Profiler_Host_EvaluateToGpuValues_Params evalParams = {
        CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
    evalParams.pHostObject = hostObj;
    evalParams.pCounterDataImage = counterDataImage.data();
    evalParams.counterDataImageSize = counterDataImage.size();
    evalParams.ppMetricNames = metric_cstrs.data();
    evalParams.numMetrics = metric_cstrs.size();
    evalParams.rangeIndex = 0;
    evalParams.pMetricValues = metricValues.data();
    check_cupti(cuptiProfilerHostEvaluateToGpuValues(&evalParams),
        "cuptiProfilerHostEvaluateToGpuValues");

    for (size_t i = 0; i < cupti_names.size(); i++) {
        result[logical_keys[i]] = metricValues[i];
        log->debug("  {} ({}) = {}", logical_keys[i], cupti_names[i], metricValues[i]);
    }

    // disable
    CUpti_RangeProfiler_Disable_Params disableParams = {
        CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE};
    disableParams.pRangeProfilerObject = rpObject;
    check_cupti(cuptiRangeProfilerDisable(&disableParams),
        "cuptiRangeProfilerDisable");

    // destroy host object so next call starts fresh
    destroy_host_object();

    return result;
}


Profiler::ProfileResult Profiler::profile(
    KernelLaunchFn launch_fn, KernelLaunchFn reset_fn)
{
    auto log = spdlog::get("profiler");

    // Step 1: Activity API (registers + shared memory)
    log->info("Collecting kernel metadata (Activity API)");
    if (reset_fn) reset_fn();
    auto result = collect_activity(launch_fn);

    // Step 2: Range Profiler (hardware counters)
    log->info("Collecting hardware counters (Range Profiler)");
    result.metric_values = collect_counters(launch_fn, reset_fn);

    return result;
}

}
