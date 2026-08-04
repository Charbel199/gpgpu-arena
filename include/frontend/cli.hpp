#pragma once

#include "arena/runner.hpp"
#include <string>
#include <vector>

namespace frontend {

// Headless benchmark runner for scripts and agents.
//
// Exit codes are load-bearing -- a caller should be able to branch on them
// without parsing output:
//   0  everything ran and verified
//   1  ran, but at least one kernel failed verification
//   2  at least one kernel errored (compile, launch, or device failure)
//   3  usage error
enum class CliExit { Ok = 0, VerificationFailed = 1, RunError = 2, UsageError = 3 };

// True when argv asks for headless mode, so main can dispatch without
// duplicating the flag list.
bool wants_cli(int argc, char** argv);

// True when headless output goes to stdout. main must consult this BEFORE it
// logs anything: otherwise startup log lines land on stdout ahead of the
// payload and the caller cannot parse it.
bool cli_writes_to_stdout(int argc, char** argv);

void print_cli_usage(const char* program);

int run_cli(arena::Runner& runner, int argc, char** argv);

}
