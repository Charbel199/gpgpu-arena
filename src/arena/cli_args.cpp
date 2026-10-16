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

std::optional<std::map<std::string, int>> parse_shape(const std::string& spec) {
    std::map<std::string, int> out;
    const auto items = split_csv(spec);
    if (items.empty()) return std::nullopt;
    for (const auto& item : items) {
        auto kv = parse_param(item);
        if (!kv || out.count(kv->first)) return std::nullopt;
        out[kv->first] = kv->second;
    }
    return out;
}

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        const size_t comma = s.find(',', start);
        const size_t end = comma == std::string::npos ? s.size() : comma;
        if (end > start) out.push_back(s.substr(start, end - start));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

std::optional<std::pair<std::string, std::vector<int>>> parse_param_list(const std::string& spec) {
    const auto eq = spec.find('=');
    if (eq == std::string::npos) return std::nullopt;
    const std::string key = spec.substr(0, eq);
    const std::string vals = spec.substr(eq + 1);
    if (key.empty() || vals.empty()) return std::nullopt;
    if (vals.front() == ',' || vals.back() == ',' || vals.find(",,") != std::string::npos)
        return std::nullopt;

    std::vector<int> out;
    for (const auto& item : split_csv(vals)) {
        auto kv = parse_param(key + "=" + item);
        if (!kv) return std::nullopt;
        out.push_back(kv->second);
    }
    if (out.empty()) return std::nullopt;
    return std::make_pair(key, out);
}

bool apply_param_list(const std::string& spec, std::map<std::string, std::vector<int>>& out) {
    auto kv = parse_param_list(spec);
    if (!kv) return false;
    out[kv->first] = kv->second;
    return true;
}

std::optional<std::vector<uint64_t>> parse_u64_list(const std::string& s) {
    std::vector<uint64_t> out;
    if (s.empty()) return std::nullopt;
    for (const auto& item : split_csv(s)) {
        char* end = nullptr;
        if (item[0] == '-') return std::nullopt;
        const unsigned long long v = std::strtoull(item.c_str(), &end, 10);
        if (end == item.c_str() || *end != '\0') return std::nullopt;
        out.push_back(static_cast<uint64_t>(v));
    }
    if (out.empty()) return std::nullopt;
    return out;
}

std::vector<std::map<std::string, int>> expand_grid(
    const std::map<std::string, std::vector<int>>& lists) {
    std::vector<std::map<std::string, int>> out = {{}};
    for (const auto& [key, values] : lists) {
        std::vector<std::map<std::string, int>> next;
        next.reserve(out.size() * values.size());
        for (const auto& partial : out) {
            for (int v : values) {
                auto m = partial;
                m[key] = v;
                next.push_back(std::move(m));
            }
        }
        out = std::move(next);
    }
    return out;
}

}
