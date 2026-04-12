#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace arena {

inline void init_logging() {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_pattern("[%H:%M:%S] [%^%l%$] [%n] %v");

    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("arena.log", true);
    file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%n] %v");

    spdlog::sinks_init_list sinks = {console_sink, file_sink};

    auto make_logger = [&](const std::string& name) {
        auto logger = std::make_shared<spdlog::logger>(name, sinks);
        spdlog::register_logger(logger);
        return logger;
    };

    auto default_logger = make_logger("arena");
    make_logger("context");
    make_logger("loader");
    make_logger("compiler");
    make_logger("profiler");
    make_logger("benchmark");
    make_logger("runner");
    make_logger("verify");

    spdlog::set_default_logger(default_logger);

#ifdef NDEBUG
    spdlog::set_level(spdlog::level::info);
#else
    spdlog::set_level(spdlog::level::debug);
#endif

    spdlog::info("Logging initialized (level={}, file=arena.log)",
#ifdef NDEBUG
        "info"
#else
        "debug"
#endif
    );
}

}
