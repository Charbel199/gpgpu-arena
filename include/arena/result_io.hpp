#pragma once

#include "arena/runner.hpp"
#include "arena/device/context.hpp"
#include "arena/measurement/power.hpp"

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

// Serialization for RunResult. Shared by the CLI and the GUI's CSV export so
// the field list exists in exactly one place.

namespace arena::result_io {

// Which language a kernel was written in.
//
// Determined by the source file, NOT by uses_module(). uses_module()==false
// only means the descriptor drives its own launches -- reduce_two_stage does
// that for its two stages while being plain CUDA C++.
std::string detect_dsl(const KernelDescriptor* d);

// Full three-tier record. Field names match the CSV columns.
nlohmann::json to_json(const RunResult& r);

// Device, driver, and toolkit provenance. Without this a result file has no
// idea what produced it, which makes cross-machine comparison guesswork.
nlohmann::json environment_json(const Context& ctx, const PowerMonitor& power);

// The config a run was executed under.
nlohmann::json config_json(const RunConfig& cfg);

// CSV. csv_header() and csv_row() must stay field-for-field aligned; the unit
// tests assert that they do.
std::string csv_header();
std::string csv_row(const RunResult& r, const std::string& dsl);

}
