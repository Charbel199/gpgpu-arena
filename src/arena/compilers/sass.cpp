#include "arena/compilers/sass.hpp"

#include <regex>
#include <sstream>

namespace arena::sass {

std::string function_text(const std::string& dump, const std::string& function) {
    const std::string marker = "Function : ";
    size_t pos = 0;
    while ((pos = dump.find(marker, pos)) != std::string::npos) {
        const size_t name_start = pos + marker.size();
        const size_t eol = dump.find('\n', name_start);
        std::string name = dump.substr(name_start, eol - name_start);
        while (!name.empty() && (name.back() == '\r' || name.back() == ' ')) name.pop_back();

        if (name == function) {
            const size_t next = dump.find(marker, name_start);
            return dump.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
        }
        pos = name_start;
    }
    return dump;
}

std::map<std::string, int> opcode_histogram(const std::string& sass) {
    // "        /*00b0*/   @!P0 LDG.E.CONSTANT R3, desc[UR4][R2.64] ;"
    static const std::regex line_re(
        R"(/\*[0-9a-fA-F]{4,}\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*))");

    std::map<std::string, int> out;
    std::istringstream in(sass);
    std::string line;
    while (std::getline(in, line)) {
        std::smatch m;
        if (!std::regex_search(line, m, line_re)) continue;
        const std::string op = m[1].str();
        if (op == "NOP") continue;
        out[op]++;
    }
    return out;
}

namespace {

std::string base_of(const std::string& op) {
    const auto dot = op.find('.');
    return dot == std::string::npos ? op : op.substr(0, dot);
}

bool has_modifier(const std::string& op, const std::string& mod) {
    const std::string needle = "." + mod;
    size_t pos = 0;
    while ((pos = op.find(needle, pos)) != std::string::npos) {
        const size_t end = pos + needle.size();
        if (end == op.size() || op[end] == '.') return true;
        pos = end;
    }
    return false;
}

const char* class_of(const std::string& op) {
    const std::string b = base_of(op);

    if (b == "LDG") {
        if (has_modifier(op, "128")) return "global_load_128";
        if (has_modifier(op, "64"))  return "global_load_64";
        return "global_load_32";
    }
    if (b == "STG") {
        if (has_modifier(op, "128")) return "global_store_128";
        if (has_modifier(op, "64"))  return "global_store_64";
        return "global_store_32";
    }
    if (b == "LDGSTS" || b == "LDGDEPBAR" || b == "UBLKCP" || b == "UTMALDG" ||
        b == "UTMASTG" || b == "UTMACCTL" || b == "UBLKRED") return "async_copy";
    if (b == "LDS" || b == "STS" || b == "LDSM" || b == "STSM") return "shared_mem";
    if (b == "LDL" || b == "STL") return "local_mem";
    if (b == "ATOMS") return "shared_atomic";
    if (b == "ATOM" || b == "ATOMG" || b == "RED" || b == "REDG" || b == "REDUX") {
        return b == "REDUX" ? "warp_reduce" : "global_atomic";
    }
    if (b == "SHFL") return "warp_shuffle";
    if (b == "BAR" || b == "MEMBAR" || b == "SYNCS" || b == "WARPSYNC" ||
        b == "BSYNC" || b == "DEPBAR") return "sync";
    if (b == "HMMA" || b == "IMMA" || b == "QMMA" || b == "OMMA" || b == "DMMA" ||
        b == "HGMMA" || b == "UTCHMMA" || b == "UTCQMMA" || b == "UTCOMMA" ||
        b == "UTCIMMA" || b == "UTCMMA" || b == "BMMA" || b == "IGMMA" ||
        b == "QGMMA" || b == "BGMMA") return "tensor";
    if (b == "FFMA" || b == "FADD" || b == "FMUL" || b == "FMNMX" || b == "FSETP" ||
        b == "FFMA2" || b == "FADD2" || b == "FMUL2" || b == "MUFU" || b == "FSEL" ||
        b == "FCHK" || b == "FRND") return "fp32";
    if (b == "HFMA2" || b == "HADD2" || b == "HMUL2" || b == "HMNMX2" || b == "HSETP2" ||
        b == "F2F" || b == "F2FP") return "fp16_and_convert";
    if (b == "DFMA" || b == "DADD" || b == "DMUL" || b == "DSETP") return "fp64";
    if (b == "BRA" || b == "EXIT" || b == "RET" || b == "CALL" || b == "BSSY" ||
        b == "JMP" || b == "BREAK" || b == "BRX") return "control";
    if (b == "IMAD" || b == "IADD3" || b == "IADD" || b == "LEA" || b == "SHF" ||
        b == "LOP3" || b == "ISETP" || b == "IMNMX" || b == "SEL" || b == "MOV" ||
        b == "IABS" || b == "SGXT" || b == "PRMT" || b == "VIADD" || b == "VIMNMX" ||
        b == "I2F" || b == "F2I" || b == "POPC" || b == "FLO" || b == "BMSK" ||
        b == "IMUL" || b == "SHL" || b == "SHR" || b == "PLOP3" || b == "P2R" ||
        b == "R2P" || b == "UIADD3" || b == "UMOV" || b == "ULEA" || b == "USHF" ||
        b == "ULOP3" || b == "UISETP" || b == "UIMAD" || b == "USEL" || b == "UPRMT" ||
        b == "S2R" || b == "S2UR" || b == "CS2R" || b == "R2UR" || b == "LDC" ||
        b == "LDCU" || b == "ULDC" || b == "VOTE" || b == "VOTEU" || b == "UFLO" ||
        b == "UPOPC" || b == "UVIADD") return "int_addr_and_misc";
    return "other";
}

}

std::map<std::string, int> classify(const std::map<std::string, int>& histogram) {
    std::map<std::string, int> out;
    for (const auto& [op, n] : histogram) out[class_of(op)] += n;
    return out;
}

double mean_global_load_bytes(const std::map<std::string, int>& histogram) {
    long bytes = 0, count = 0;
    for (const auto& [op, n] : histogram) {
        if (base_of(op) != "LDG") continue;
        int w = 4;
        if (has_modifier(op, "128")) w = 16;
        else if (has_modifier(op, "64")) w = 8;
        else if (has_modifier(op, "U16") || has_modifier(op, "S16")) w = 2;
        else if (has_modifier(op, "U8") || has_modifier(op, "S8")) w = 1;
        bytes += static_cast<long>(w) * n;
        count += n;
    }
    return count ? static_cast<double>(bytes) / count : 0.0;
}

ResourceUsage parse_resource_usage(const std::string& dump, const std::string& function) {
    // " Function reduce_sum_baseline:\n  REG:8 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:916 ..."
    static const std::regex fn_re(R"(Function\s+([^:\s]+):\s*\n\s*(REG:[^\n]*))");
    auto field = [](const std::string& line, const std::string& key) {
        std::regex re(key + R"(:(\d+))");
        std::smatch m;
        return std::regex_search(line, m, re) ? std::stoi(m[1].str()) : 0;
    };

    ResourceUsage first, named;
    for (auto it = std::sregex_iterator(dump.begin(), dump.end(), fn_re);
         it != std::sregex_iterator(); ++it) {
        const std::string name = (*it)[1].str();
        const std::string line = (*it)[2].str();
        ResourceUsage r;
        r.found = true;
        r.registers = field(line, "REG");
        r.stack = field(line, "STACK");
        r.shared = field(line, "SHARED");
        r.local = field(line, "LOCAL");
        r.constant0 = field(line, R"(CONSTANT\[0\])");
        if (!first.found) first = r;
        if (name == function) named = r;
    }
    return named.found ? named : first;
}

}
