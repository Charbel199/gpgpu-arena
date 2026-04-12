#include "arena/compilers/cutile_compiler.hpp"
#include "arena/compilers/compiler_utils.hpp"

namespace arena {

CuTileCompiler::CuTileCompiler(const std::string& kernel_dir)
    : kernel_dir_(kernel_dir) {}

CompileResult CuTileCompiler::compile(const std::string& source_path,
                                       const std::string& output_name,
                                       const std::string& cache_dir) {
    std::string cmd =
        "PYTHONPATH=" + kernel_dir_ +
        " " ARENA_PYTHON " " + kernel_dir_ + "/" + source_path +
        " --output-dir " + cache_dir +
        " --output-name " + output_name +
        " 2>&1";

    auto output = run_command(cmd, "cuTile compiler for " + source_path);
    auto j = parse_json_output(output, "cuTile compiler for " + source_path);
    return result_from_json(j, cache_dir + "/" + output_name + ".cubin");
}

}
