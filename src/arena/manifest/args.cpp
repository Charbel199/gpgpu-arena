#include "arena/manifest/args.hpp"
#include "arena/warp_abi.hpp"

#include <cstring>
#include <stdexcept>

namespace arena::manifest {

namespace {

uint64_t buffer(const ArgContext& ctx, const std::string& name) {
    auto it = ctx.buffers.find(name);
    if (it == ctx.buffers.end()) {
        std::string known;
        for (const auto& [k, v] : ctx.buffers) known += (known.empty() ? "" : ", ") + k;
        throw std::runtime_error("unknown buffer '" + name + "' (have: " + known + ")");
    }
    return it->second;
}

std::vector<int64_t> eval_list(const nlohmann::json& j, const Vars& vars, const char* what) {
    if (!j.is_array() || j.empty() || j.size() > 4) {
        throw std::runtime_error(std::string(what) + " must be a list of 1 to 4 sizes");
    }
    std::vector<int64_t> out;
    for (const auto& v : j) out.push_back(eval_json(v, vars));
    return out;
}

void push_cutile_array(PackedArgs& out, const nlohmann::json& a, const ArgContext& ctx,
                       nlohmann::json& said) {
    const uint64_t ptr = buffer(ctx, a.at("cutile_array").get<std::string>());
    const auto shape = eval_list(a.at("shape"), ctx.vars, "cutile_array shape");
    std::vector<int64_t> strides;
    if (a.contains("strides")) {
        strides = eval_list(a.at("strides"), ctx.vars, "cutile_array strides");
        if (strides.size() != shape.size())
            throw std::runtime_error("cutile_array strides must match shape in length");
    } else {
        strides.assign(shape.size(), 1);
        for (int d = (int)shape.size() - 2; d >= 0; d--) strides[d] = strides[d + 1] * shape[d + 1];
    }
    out.push(ptr);
    for (auto s : shape)   out.push(static_cast<uint32_t>(s));
    for (auto s : strides) out.push(static_cast<uint32_t>(s));
    said = {{"cutile_array", a.at("cutile_array")}, {"shape", shape}, {"strides", strides}};
}

void push_warp_array(PackedArgs& out, const nlohmann::json& a, const ArgContext& ctx,
                     nlohmann::json& said) {
    const uint64_t ptr = buffer(ctx, a.at("warp_array").get<std::string>());
    const auto shape = eval_list(a.at("shape"), ctx.vars, "warp_array shape");
    WarpArrayF32 arr{};
    arr.data = ptr;
    arr.ndim = static_cast<int32_t>(shape.size());
    for (size_t d = 0; d < shape.size(); d++) arr.shape[d] = static_cast<int32_t>(shape[d]);
    if (a.contains("strides")) {
        const auto st = eval_list(a.at("strides"), ctx.vars, "warp_array strides");
        if (st.size() != shape.size())
            throw std::runtime_error("warp_array strides must match shape in length");
        for (size_t d = 0; d < st.size(); d++) arr.strides[d] = static_cast<int32_t>(st[d]);
    } else {
        int64_t s = sizeof(float);
        for (int d = (int)shape.size() - 1; d >= 0; d--) {
            arr.strides[d] = static_cast<int32_t>(s);
            s *= shape[d];
        }
    }
    out.push(arr);
    said = {{"warp_array", a.at("warp_array")},
            {"shape", std::vector<int32_t>(arr.shape, arr.shape + shape.size())},
            {"strides", std::vector<int32_t>(arr.strides, arr.strides + shape.size())}};
}

// A scalar of one width, with the expression it came from.
template <typename T>
void push_scalar(PackedArgs& out, const char* type, const nlohmann::json& expr,
                 const ArgContext& ctx, nlohmann::json& said) {
    const T v = static_cast<T>(eval_json(expr, ctx.vars));
    out.push(v);
    said = {{"type", type}, {"expr", expr}, {"value", v}};
}

}

PackedArgs pack_args(const nlohmann::json& args, const ArgContext& ctx, nlohmann::json* described) {
    if (!args.is_array()) throw std::runtime_error("\"args\" must be a list");
    PackedArgs out;
    if (described) *described = nlohmann::json::array();
    for (size_t i = 0; i < args.size(); i++) {
        const auto& a = args[i];
        nlohmann::json said;
        try {
            if (a.is_string()) {
                const std::string s = a.get<std::string>();
                if (s == "null") {
                    out.push(uint64_t{0});
                    said = {{"null", true}};
                } else if (s == "...") {
                    if (i + 1 != args.size()) throw std::runtime_error("\"...\" must be last");
                    const size_t before = out.size();
                    while (out.size() < ctx.param_sizes.size()) out.push(uint64_t{0});
                    said = {{"padding", out.size() - before}};
                } else if (ctx.buffers.count(s)) {
                    out.push(ctx.buffers.at(s));
                    said = {{"buffer", s}};
                } else if (ctx.vars.count(s)) {
                    const auto v = static_cast<int32_t>(ctx.vars.at(s));
                    out.push(v);
                    said = {{"name", s}, {"type", "i32"}, {"value", v}};
                } else {
                    throw std::runtime_error("unknown name '" + s + "'");
                }
            } else if (a.is_object() && a.size() >= 1) {
                if (a.contains("i32"))           push_scalar<int32_t>(out, "i32", a.at("i32"), ctx, said);
                else if (a.contains("u32"))      push_scalar<uint32_t>(out, "u32", a.at("u32"), ctx, said);
                else if (a.contains("i64"))      push_scalar<int64_t>(out, "i64", a.at("i64"), ctx, said);
                else if (a.contains("cutile_array")) push_cutile_array(out, a, ctx, said);
                else if (a.contains("warp_array"))   push_warp_array(out, a, ctx, said);
                else if (a.contains("warp_bounds")) {
                    const auto n = static_cast<int32_t>(eval_json(a.at("warp_bounds"), ctx.vars));
                    out.push(warp_bounds_1d(n));
                    said = {{"warp_bounds", n}};
                }
                else throw std::runtime_error("unknown argument kind " + a.dump());
            } else {
                throw std::runtime_error("unsupported argument " + a.dump());
            }
            if (described) described->push_back(std::move(said));
        } catch (const nlohmann::json::exception& e) {
            throw std::runtime_error("argument " + std::to_string(i) + ": " + e.what());
        } catch (const std::runtime_error& e) {
            throw std::runtime_error("argument " + std::to_string(i) + ": " + e.what());
        }
    }
    if (!ctx.param_sizes.empty()) {
        if (out.size() != ctx.param_sizes.size()) {
            throw std::runtime_error("the kernel takes " + std::to_string(ctx.param_sizes.size()) +
                                     " parameters but the manifest passes " + std::to_string(out.size()));
        }
        for (size_t i = 0; i < out.size(); i++) {
            if (out.size_of(i) != ctx.param_sizes[i]) {
                throw std::runtime_error("argument " + std::to_string(i) + " is " +
                    std::to_string(out.size_of(i)) + " bytes but the kernel's parameter " +
                    std::to_string(i) + " is " + std::to_string(ctx.param_sizes[i]) + " bytes");
            }
        }
    }
    return out;
}

}
