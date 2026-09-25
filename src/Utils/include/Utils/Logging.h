#pragma once

#include "Diagnostics.h"

#include <spdlog/spdlog.h>

#include <string>
#include <span>

#define __SHORT_FILE__ __FILE__
#if 1

#    undef __SHORT_FILE__
#    define __SHORT_FILE__   past_last_slash(__FILE__)
/*
        ({                                                       \
            constexpr const char* p = past_last_slash(__FILE__); \
            p;                                                   \
        })
*/
static constexpr const char* past_last_slash(const char* const path, const int pos = 0,
                                             const int last_slash = 0) {
    if (path[pos] == '\0') return &path[last_slash];
    if (path[pos] == '/')
        return past_last_slash(path, pos + 1, pos + 1);
    else
        return past_last_slash(path, pos + 1, last_slash);
}

#endif

enum
{
    LOGLEVEL_INFO  = 0,
    LOGLEVEL_WARN  = 1,
    LOGLEVEL_ERROR = 2
};

// spdlog's compiled level removes the complete INFO call, including printf formatting and
// argument evaluation, from production. Keep the printf-facing API at this boundary so scene
// and GPU code only describe their messages. Authored console output uses WallpaperLog and
// stays observable independently of the renderer's internal diagnostic level.
#define LOG_INFO(...) \
    SPDLOG_LOGGER_INFO(WallpaperLogger(), "{}", \
        FormatWallpaperLog(LOGLEVEL_INFO, "", 0, __VA_ARGS__))
#define LOG_WARN(...) \
    SPDLOG_LOGGER_WARN(WallpaperLogger(), "{}", \
        FormatWallpaperLog(LOGLEVEL_WARN, __SHORT_FILE__, __LINE__, __VA_ARGS__))
#define LOG_ERROR(...) \
    SPDLOG_LOGGER_ERROR(WallpaperLogger(), "{}", \
        FormatWallpaperLog(LOGLEVEL_ERROR, __SHORT_FILE__, __LINE__, __VA_ARGS__))

spdlog::logger* WallpaperLogger();
std::string FormatWallpaperLog(int level, const char* file, int line, const char* fmt, ...);
void WallpaperLog(int level, const char* file, int line, const char* fmt, ...);

std::string logToTmpfileWithSha1(std::span<const char>, const char* fmt, ...);
