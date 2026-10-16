#pragma once

#include <string>

// Resolving paths the user typed or pasted, for --open, --manifest,
// --source-override, --artifacts-dir and the GUI's Open box.

namespace arena {

// Drops surrounding whitespace and quotes, which pasted paths often carry.
std::string trim_path(const std::string& given);

// A relative path is taken from the current directory when it exists there,
// else from `root` (the repo root) when it exists there, else from the
// current directory. So a path works from wherever arena is started, and
// also from inside a build folder or Docker where the two differ. The result
// is absolute.
std::string resolve_user_path(const std::string& given, const std::string& root);

}
