#include "arena/cli_args.hpp"

#include <cstdlib>

namespace arena::cli {

std::optional<std::pair<std::string, int>> parse_param(const std::string& spec) {
    const auto eq = spec.find('=');
    if (eq == std::string::npos) return std::nullopt;

    const std::string key = spec.substr(0, eq);
    const std::string val = spec.substr(eq + 1);
    if (key.empty() || val.empty()) return std::nullopt;

    char* end = nullptr;
    const long parsed = std::strtol(val.c_str(), &end, 10);
    if (end == val.c_str() || *end != '\0') return std::nullopt;   // junk or empty
    if (parsed < 0) return std::nullopt;

    return std::make_pair(key, static_cast<int>(parsed));
}

bool apply_params(const std::string& spec, std::map<std::string, int>& out) {
    auto kv = parse_param(spec);
    if (!kv) return false;
    out[kv->first] = kv->second;
    return true;
}

std::vector<TuningVariant> tuning_variants_for(
    bool sweep,
    int requested_block,
    const std::map<std::string, int>& requested_defines,
    const std::vector<int>& tunable_blocks,
    const std::vector<std::map<std::string, int>>& tunable_defines) {

    if (sweep && !tunable_blocks.empty()) {
        std::vector<TuningVariant> out;
        out.reserve(tunable_blocks.size());
        for (int b : tunable_blocks) out.push_back({b, requested_defines});
        return out;
    }
    if (sweep && !tunable_defines.empty()) {
        std::vector<TuningVariant> out;
        out.reserve(tunable_defines.size());
        for (const auto& d : tunable_defines) out.push_back({requested_block, d});
        return out;
    }
    return {{requested_block, requested_defines}};
}

}
