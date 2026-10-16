#include "arena/paths.hpp"

#include <cctype>
#include <filesystem>

namespace fs = std::filesystem;

namespace arena {

std::string trim_path(const std::string& given) {
    const auto junk = [](char c) {
        return std::isspace(static_cast<unsigned char>(c)) || c == '"' || c == '\'';
    };
    size_t b = 0, e = given.size();
    while (b < e && junk(given[b])) b++;
    while (e > b && junk(given[e - 1])) e--;
    return given.substr(b, e - b);
}

std::string resolve_user_path(const std::string& given, const std::string& root) {
    const fs::path p(trim_path(given));
    if (p.empty() || p.is_absolute()) return p.lexically_normal().string();
    std::error_code ec;
    const fs::path here = fs::current_path(ec) / p;
    if (fs::exists(here, ec)) return here.lexically_normal().string();
    const fs::path there = fs::path(root) / p;
    if (!root.empty() && fs::exists(there, ec)) return there.lexically_normal().string();
    return here.lexically_normal().string();
}

}
