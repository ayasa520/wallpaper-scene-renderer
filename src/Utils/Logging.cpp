#include "Logging.h"
#include <cstdio>
#include <cstdarg>
#include <filesystem>
#include <memory>

#include <spdlog/sinks/stdout_sinks.h>

#include "Sha.hpp"

constexpr const char* level_names[] = { "INFO", "WARN", "ERROR" };
constexpr spdlog::level::level_enum log_levels[] = {
    spdlog::level::info, spdlog::level::warn, spdlog::level::err
};

namespace
{

std::string FormatMessage(int level, const char* file, int line,
                          const char* format, std::va_list args) {
    std::string result = level == LOGLEVEL_INFO
        ? fmt::format("{:<5}", level_names[level])
        : fmt::format("{:<5} {}:{} ", level_names[level], file, line);
    std::va_list length_args;
    va_copy(length_args, args);
    const int length = std::vsnprintf(nullptr, 0, format, length_args);
    va_end(length_args);
    if (length <= 0) return result;
    const auto prefix_size = result.size();
    result.resize(prefix_size + static_cast<size_t>(length));
    std::vsnprintf(result.data() + prefix_size, static_cast<size_t>(length) + 1, format, args);
    return result;
}

} // namespace

spdlog::logger* WallpaperLogger() {
    // One thread-safe stderr sink keeps a diagnostic record intact across the loading, audio
    // and rendering threads. Compile-time levels select internal records; the logger itself
    // also accepts authored console output in production and flushes records immediately.
    static const auto logger = [] {
        auto result = std::make_unique<spdlog::logger>(
            "scene", std::make_shared<spdlog::sinks::stderr_sink_mt>());
        result->set_pattern("%v");
        result->flush_on(spdlog::level::info);
        return result;
    }();
    return logger.get();
}

std::string FormatWallpaperLog(int level, const char* file, int line, const char* format, ...) {
    std::va_list args;
    va_start(args, format);
    auto result = FormatMessage(level, file, line, format, args);
    va_end(args);
    return result;
}

void WallpaperLog(int level, const char* file, int line, const char* fmt, ...) {
    std::va_list args;
    va_start(args, fmt);
    auto message = FormatMessage(level, file, line, fmt, args);
    va_end(args);
    WallpaperLogger()->log(log_levels[level], "{}", message);
}

std::string logToTmpfileWithSha1(std::span<const char> in, const char* fmt, ...) {
    std::va_list          args;
    std::string           name   = utils::genSha1(in);
    std::filesystem::path fspath = std::filesystem::temp_directory_path() / name;
    std::string           path   = fspath.native();
    auto*                 file   = std::fopen(path.c_str(), "w+");
    {
        va_start(args, fmt);
        std::vfprintf(file, fmt, args);
        va_end(args);
    }
    std::fprintf(file, "\n");
    std::fclose(file);
    return path;
}
