#include "arena/compilers/python_backend.hpp"
#include "arena/compilers/compiler_utils.hpp"
#include <filesystem>

namespace arena {

PythonBackend::PythonBackend(const std::string& kernel_dir, std::string label,
                             std::string package, std::string base_module)
    : kernel_dir_(kernel_dir), label_(std::move(label)),
      package_(std::move(package)), base_module_(std::move(base_module)) {}

std::string PythonBackend::toolchain_id() {
    if (toolchain_.empty()) {
        std::string version;
        try {
            version = run_command(
                std::string(ARENA_PYTHON) +
                " -c \"import importlib.metadata as m; print(m.version('" + package_ + "'))\" 2>&1",
                package_ + " version");
            while (!version.empty() && (version.back() == '\n' || version.back() == '\r'))
                version.pop_back();
        } catch (...) {
            // Not installed. The compile will fail with a clearer message, and
            // a later install changes this string, so nothing stale survives.
            version = "missing";
        }
        toolchain_ = package_ + " " + version + " " + ARENA_PYTHON;
    }
    return toolchain_;
}

std::vector<std::string> PythonBackend::dependencies() const {
    return {kernel_dir_ + "/" + base_module_};
}

CompileResult PythonBackend::compile(const std::string& source_path,
                                     const std::string& output_name,
                                     const std::string& cache_dir,
                                     const CompileDefines& defines) {
    std::string cmd =
        "PYTHONPATH=" + shell_quote(kernel_dir_) +
        " " ARENA_PYTHON " " + shell_quote(resolve_source(kernel_dir_, source_path)) +
        " --output-dir " + shell_quote(cache_dir) +
        " --output-name " + output_name +
        defines_args(defines) +
        " 2>&1";

    const std::string context = label_ + " compiler for " + source_path;
    auto output = run_command(cmd, context);
    auto j = parse_json_output(output, context);
    return result_from_json(j, cache_dir + "/" + output_name + ".cubin");
}

std::string PythonBackend::emit_ptx(const std::string& source_path,
                                    const std::string& output_name,
                                    const std::string& cache_dir,
                                    const CompileDefines& defines) {
    (void)source_path; (void)defines;
    const auto ptx = cache_dir + "/" + output_name + ".ptx";
    return std::filesystem::exists(ptx) ? ptx : "";
}

}
