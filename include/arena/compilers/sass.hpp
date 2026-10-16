#pragma once

#include <map>
#include <string>
#include <vector>

// Parsing for cuobjdump output. Pure text in, structs out, so it is
// unit-tested against captured dumps without a GPU or a toolkit.

namespace arena::sass {

// The SASS of one function from a `cuobjdump -sass` dump, or the whole dump
// when the function is not found (a cubin with one kernel and a mangled name).
std::string function_text(const std::string& dump, const std::string& function);

// Instruction mnemonics with their modifiers ("LDG.E.128.CONSTANT"), counted.
// Predicates are stripped. NOPs are left out: they are scheduling padding,
// not work, and they swamp the counts.
std::map<std::string, int> opcode_histogram(const std::string& sass);

// Counts grouped into what a performance reading cares about: how memory is
// touched (and how wide), how threads cooperate, and what does the math.
// Every instruction lands in exactly one class, "other" included.
std::map<std::string, int> classify(const std::map<std::string, int>& histogram);

// Bytes moved per global load instruction, averaged over all LDGs. 4 for
// scalar fp32 loads, 16 for float4. 0 when there are no LDGs.
double mean_global_load_bytes(const std::map<std::string, int>& histogram);

struct ResourceUsage {
    bool found = false;
    int registers = 0;
    int stack = 0;
    int shared = 0;
    int local = 0;       // > 0 means spills or a local array
    int constant0 = 0;
};

// From `cuobjdump -res-usage`. Falls back to the first function listed when
// the named one is absent.
ResourceUsage parse_resource_usage(const std::string& dump, const std::string& function);

}
