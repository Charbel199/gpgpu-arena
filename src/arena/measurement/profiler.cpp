#include "arena/measurement/profiler.hpp"
#include "arena/device/utils.hpp"
#include <spdlog/spdlog.h>
#include <cstdlib>
#include <unordered_map>
#include <string>

namespace arena {

// Note: Profiler (CUPTI) will not run if nsys or ncu used (They will reserve CUPTI context before the arena app)

// One record per launch. Keyed by name this used to overwrite: an operation
// that launched the same kernel twice reported one launch with the second
// one's duration, and the map lost the order the launches ran in.
static std::vector<KernelLaunchRecord> s_activity_records;

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
            KernelLaunchRecord r;
            r.name = kernel->name ? kernel->name : "";
            r.start_ns = kernel->start;
            r.end_ns = kernel->end;
            r.registers_per_thread = kernel->registersPerThread;
            r.shared_memory = kernel->staticSharedMemory + kernel->dynamicSharedMemory;
            s_activity_records.push_back(std::move(r));
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

    auto log = spdlog::get("profiler");
    auto result = cuptiActivityRegisterCallbacks(
        activity_buffer_requested, activity_buffer_completed);
    if (result != CUPTI_SUCCESS) {
        log->warn("CUPTI Activity API registration failed (nsys/ncu running?)");
        return;
    }

    log->info("CUPTI Activity API initialized");
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
    destroy_host_object();   // one left by a call that threw
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


// Candidate CUPTI names per logical metric, first match wins. Pipe names
// differ between generations (gb202 has no sm__pipe_lsu_cycles_active or
// fp16 pipe, for example), so the instruction-issue form is the fallback.
namespace {
struct FullMetricSpec {
    std::string key;
    std::vector<std::string> candidates;
    bool additive;
};

std::vector<FullMetricSpec> full_metric_specs() {
    const std::string pct = ".avg.pct_of_peak_sustained_active";
    auto pipe = [&](const std::string& name, std::vector<std::string> bases) {
        std::vector<std::string> c;
        for (const auto& b : bases) c.push_back(b + pct);
        return FullMetricSpec{std::string(metric::PIPE_PREFIX) + name, c, false};
    };

    std::vector<FullMetricSpec> specs = {
        {metric::ISSUE_ACTIVE,   {"smsp__issue_active" + pct}, false},
        {metric::ELIGIBLE_WARPS, {"smsp__warps_eligible.avg.per_cycle_active"}, false},
        pipe("fma",    {"sm__pipe_fma_cycles_active", "sm__inst_executed_pipe_fma"}),
        pipe("alu",    {"sm__pipe_alu_cycles_active", "sm__inst_executed_pipe_alu"}),
        pipe("lsu",    {"sm__pipe_lsu_cycles_active", "sm__inst_executed_pipe_lsu"}),
        pipe("fp16",   {"sm__pipe_fp16_cycles_active", "sm__inst_executed_pipe_fp16",
                        "sm__inst_executed_pipe_fma_type_fp16"}),
        pipe("tensor", {"sm__pipe_tensor_cycles_active", "sm__pipe_tensor_op_hmma_cycles_active",
                        "sm__inst_executed_pipe_tensor"}),
        pipe("xu",     {"sm__pipe_xu_cycles_active", "sm__inst_executed_pipe_xu"}),
        {metric::SHARED_WAVEFRONTS, {"l1tex__data_pipe_lsu_wavefronts_mem_shared.sum"}, true},
        {metric::SHARED_BANK_CONFLICTS, {"l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"}, true},
        {metric::L2_SECTORS, {"lts__t_sectors.sum"}, true},
        {metric::GPU_TIME_NS, {"gpu__time_duration.sum"}, true},
    };

    for (const char* reason : {"barrier", "branch_resolving", "dispatch_stall", "drain",
                               "imc_miss", "lg_throttle", "long_scoreboard",
                               "math_pipe_throttle", "membar", "mio_throttle", "misc",
                               "no_instruction", "not_selected", "selected",
                               "short_scoreboard", "sleeping", "tex_throttle", "wait"}) {
        specs.push_back({std::string(metric::STALL_PREFIX) + reason,
            {std::string("smsp__average_warps_issue_stalled_") + reason +
             "_per_issue_active.ratio"}, false});
    }
    return specs;
}
}

void Profiler::resolve_full_metrics() {
    if (full_metrics_resolved_) return;
    auto log = spdlog::get("profiler");
    full_metrics_.clear();   // a probe that threw part way is redone

    // Each candidate is tried on its own in a scratch host object: one
    // unknown name fails the whole config, so the set has to be filtered
    // before it is used. A failed add gets a fresh object in case it left
    // the old one in a bad state.
    bool need_fresh = true;
    for (const auto& spec : full_metric_specs()) {
        bool found = false;
        for (const auto& name : spec.candidates) {
            if (need_fresh) {
                destroy_host_object();
                create_host_object();
                need_fresh = false;
            }
            const char* cname = name.c_str();
            CUpti_Profiler_Host_ConfigAddMetrics_Params p = {
                CUpti_Profiler_Host_ConfigAddMetrics_Params_STRUCT_SIZE};
            p.pHostObject = host_object_;
            p.ppMetricNames = &cname;
            p.numMetrics = 1;
            if (cuptiProfilerHostConfigAddMetrics(&p) == CUPTI_SUCCESS) {
                full_metrics_.push_back({spec.key, name, spec.additive});
                found = true;
                break;
            }
            need_fresh = true;
            log->debug("counters full: {} not available on {}", name, chip_name_);
        }
        if (!found) log->debug("counters full: dropping {}", spec.key);
    }
    destroy_host_object();

    full_metrics_resolved_ = true;
    log->info("counters full: {} extra metrics available on {}",
        full_metrics_.size(), chip_name_);
}

Profiler::ProfileResult Profiler::collect_activity(KernelLaunchFn launch_fn) {
    s_activity_records.clear();
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);

    launch_fn();
    cuCtxSynchronize();

    cuptiActivityFlushAll(0);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);

    auto result = summarize_launches(s_activity_records);

    auto log = spdlog::get("profiler");
    for (const auto& sk : result.sub_kernels) {
        log->debug("  GPU kernel: {} | {:.3f} ms | {} regs | {} B shmem",
            sk.name, sk.duration_ms, sk.registers, sk.shared_memory);
    }
    if (result.sub_kernels.size() > 1) {
        log->debug("  GPU total: {:.3f} ms ({} launches)",
            result.kernel_time_ms, result.sub_kernels.size());
    }
    return result;
}

// Launches per counter pass. AutoRange gives one range per launch, so this
// runs a short back-to-back sequence and reads the last one. A single isolated
// launch arrives with a cold L2, because profiler setup evicts whatever the
// timed loop left there, and then reports the whole working set as DRAM
// traffic even when the real loop serves most of it from cache. Five is enough
// for a streaming kernel to reach the same cache state the timed loop runs in.
static constexpr int kCounterWarmupLaunches = 12;

std::map<std::string, double> Profiler::collect_counters(
    KernelLaunchFn launch_fn, KernelLaunchFn reset_fn, int launches_per_op, bool full)
{
    auto log = spdlog::get("profiler");
    std::map<std::string, double> result;
    if (launches_per_op < 1) launches_per_op = 1;
    const int max_ranges = kCounterWarmupLaunches * launches_per_op;

    init_range_profiler();

    // CUPTI metric names for this chip (set in init_range_profiler)
    std::vector<std::string> cupti_names = {
        chip_metrics_.dram_read, chip_metrics_.dram_write,
        chip_metrics_.occupancy, chip_metrics_.ipc
    };
    // Corresponding logical keys returned to caller
    std::vector<std::string> logical_keys = {
        metric::DRAM_READ, metric::DRAM_WRITE,
        metric::OCCUPANCY, metric::IPC
    };
    // Bytes add up across an operation's launches, rates do not. DRAM read
    // stays first: select_counter_op keys on metric 0.
    std::vector<bool> additive = {true, true, false, false};
    // Rates come from the operation's longest launch when its duration is
    // measured, otherwise from the one with the most DRAM reads.
    size_t dominant_metric = 0;

    if (full) {
        resolve_full_metrics();
        for (const auto& m : full_metrics_) {
            if (m.key == metric::GPU_TIME_NS) dominant_metric = cupti_names.size();
            cupti_names.push_back(m.cupti_name);
            logical_keys.push_back(m.key);
            additive.push_back(m.additive);
        }
    }

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

    // On an early exit (a CUPTI error, a failed launch, no complete
    // operation) the profiler is disabled here, or the next call's enable
    // fails on this context and every later kernel loses its counters.
    struct EnabledGuard {
        Profiler* self;
        decltype(rpObject) rp;
        bool armed = true;
        ~EnabledGuard() {
            if (!armed) return;
            CUpti_RangeProfiler_Disable_Params p = {
                CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE};
            p.pRangeProfilerObject = rp;
            cuptiRangeProfilerDisable(&p);
            self->destroy_host_object();
        }
    } guard{this, rpObject};

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
    last_num_passes_ = static_cast<int>(numPassesParams.numOfPasses);
    log->debug("Range Profiler: {} passes for {} metrics",
        numPassesParams.numOfPasses, cupti_names.size());

    // counter data image
    CUpti_RangeProfiler_GetCounterDataSize_Params counterDataSizeParams = {
        CUpti_RangeProfiler_GetCounterDataSize_Params_STRUCT_SIZE};
    counterDataSizeParams.pRangeProfilerObject = rpObject;
    counterDataSizeParams.pMetricNames = metric_cstrs.data();
    counterDataSizeParams.numMetrics = metric_cstrs.size();
    counterDataSizeParams.maxNumOfRanges = max_ranges;
    counterDataSizeParams.maxNumRangeTreeNodes = max_ranges;
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
    setConfigParams.maxRangesPerPass = max_ranges;
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

        // reset before each launch, exactly as the timed loop does, so the
        // cache state these counters see is the one being timed.
        for (int i = 0; i < kCounterWarmupLaunches; i++) {
            if (reset_fn) reset_fn();
            launch_fn();
        }
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
    // Every range is read, then grouped back into operations by
    // select_counter_op. The profiler re-cools L2 every few ranges, so a given
    // launch is warm or cold depending on where it lands in that cycle: a
    // 12-range trace of one kernel read 86, 4.7, 0.07, 0.05, 0.00, 86, 4.0,
    // 2.4, 0.001, 85, 4.1, 0.05 MB. Instrumentation can only add DRAM traffic,
    // so the operation with the least is the closest estimate of the
    // untouched timed loop. Bytes add up across an operation's launches;
    // occupancy and IPC come from its dominant launch.
    std::vector<std::vector<double>> ranges;
    for (int ri = 0; ri < max_ranges; ri++) {
        std::vector<double> v(cupti_names.size(), 0.0);
        CUpti_Profiler_Host_EvaluateToGpuValues_Params ep = {
            CUpti_Profiler_Host_EvaluateToGpuValues_Params_STRUCT_SIZE};
        ep.pHostObject = hostObj;
        ep.pCounterDataImage = counterDataImage.data();
        ep.counterDataImageSize = counterDataImage.size();
        ep.ppMetricNames = metric_cstrs.data();
        ep.numMetrics = metric_cstrs.size();
        ep.rangeIndex = ri;
        ep.pMetricValues = v.data();
        if (cuptiProfilerHostEvaluateToGpuValues(&ep) != CUPTI_SUCCESS) break;
        ranges.push_back(std::move(v));
    }
    log->debug("Range Profiler: {} ranges, {} launches per operation",
        ranges.size(), launches_per_op);

    auto metricValues = select_counter_op(ranges, launches_per_op, additive, dominant_metric);
    if (metricValues.empty()) {
        throw std::runtime_error("Range Profiler returned no complete operation");
    }

    for (size_t i = 0; i < cupti_names.size(); i++) {
        result[logical_keys[i]] = metricValues[i];
        log->debug("  {} ({}) = {}", logical_keys[i], cupti_names[i], metricValues[i]);
    }

    // disable
    guard.armed = false;
    CUpti_RangeProfiler_Disable_Params disableParams = {
        CUpti_RangeProfiler_Disable_Params_STRUCT_SIZE};
    disableParams.pRangeProfilerObject = rpObject;
    check_cupti(cuptiRangeProfilerDisable(&disableParams),
        "cuptiRangeProfilerDisable");

    // destroy host object so next call starts fresh
    destroy_host_object();

    return result;
}


}
