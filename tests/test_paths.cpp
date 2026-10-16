#include <doctest/doctest.h>
#include "arena/paths.hpp"

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;
using arena::resolve_user_path;
using arena::trim_path;

TEST_CASE("trim_path") {
    CHECK(trim_path("pgaio/runs/scan-1 ") == "pgaio/runs/scan-1");
    CHECK(trim_path("  \"/a b/c\"\n") == "/a b/c");
    CHECK(trim_path("'x'") == "x");
    CHECK(trim_path("   ") == "");
}

TEST_CASE("resolve_user_path") {
    const fs::path base = fs::temp_directory_path() / "arena_test_paths";
    fs::remove_all(base);
    fs::create_directories(base / "cwd" / "only_here");
    fs::create_directories(base / "root" / "only_root");
    fs::create_directories(base / "cwd" / "both");
    fs::create_directories(base / "root" / "both");
    const fs::path saved = fs::current_path();
    fs::current_path(base / "cwd");
    const std::string root = (base / "root").string();

    CHECK(resolve_user_path("only_here", root) == (base / "cwd" / "only_here").string());
    CHECK(resolve_user_path("only_root", root) == (base / "root" / "only_root").string());
    CHECK(resolve_user_path("both", root) == (base / "cwd" / "both").string());
    // A root that does not exist on this machine (a binary built in Docker).
    CHECK(resolve_user_path("only_here", "/no/such/root") == (base / "cwd" / "only_here").string());
    // Missing everywhere: from the current directory, so errors name it.
    CHECK(resolve_user_path("missing", root) == (base / "cwd" / "missing").string());
    CHECK(resolve_user_path("/abs/./x ", root) == "/abs/x");

    fs::current_path(saved);
    fs::remove_all(base);
}
