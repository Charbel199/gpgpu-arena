#include "arena/manifest/expr.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <vector>

namespace arena::manifest {

namespace {

class Parser {
public:
    Parser(const std::string& s, const Vars& vars) : s_(s), vars_(vars) {}

    int64_t parse() {
        const int64_t v = conditional();
        skip_ws();
        if (pos_ != s_.size()) fail("unexpected '" + std::string(1, s_[pos_]) + "'");
        return v;
    }

private:
    [[noreturn]] void fail(const std::string& why) const {
        throw std::runtime_error("expression \"" + s_ + "\": " + why);
    }

    void skip_ws() {
        while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) pos_++;
    }

    bool eat(char c) {
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == c) { pos_++; return true; }
        return false;
    }

    // Division by zero in a branch that is not taken is not an error, so
    // "M > 0 ? N / M : 0" works at M = 0. Syntax errors and unknown names
    // still are, wherever they are.
    static int64_t div_checked(int64_t a, int64_t b, const Parser& p) {
        if (b == 0) {
            if (p.skip_) return 0;
            p.fail("division by zero");
        }
        return a / b;
    }

    // Parses an operand without letting it fail on arithmetic, for the side
    // of ?:, && or || that the condition rules out.
    template <typename F>
    int64_t operand(bool taken, F f) {
        if (taken) return f();
        skip_++;
        f();
        skip_--;
        return 0;
    }

    // c ? a : b, lowest precedence and right-associative, as in C.
    int64_t conditional() {
        const int64_t c = logical_or();
        if (!eat('?')) return c;
        const int64_t a = operand(c != 0, [&] { return conditional(); });
        if (!eat(':')) fail("missing ':' after the '?' branch");
        const int64_t b = operand(c == 0, [&] { return conditional(); });
        return c ? a : b;
    }

    // Comparisons and logic give 1 or 0, for a launch's "when". && and ||
    // short-circuit like C.
    int64_t logical_or() {
        int64_t v = logical_and();
        while (eat2("||")) {
            const int64_t r = operand(v == 0, [&] { return logical_and(); });
            v = (v || r) ? 1 : 0;
        }
        return v;
    }

    int64_t logical_and() {
        int64_t v = comparison();
        while (eat2("&&")) {
            const int64_t r = operand(v != 0, [&] { return comparison(); });
            v = (v && r) ? 1 : 0;
        }
        return v;
    }

    int64_t comparison() {
        int64_t v = sum();
        for (;;) {
            if (eat2("<="))      v = v <= sum();
            else if (eat2(">=")) v = v >= sum();
            else if (eat2("==")) v = v == sum();
            else if (eat2("!=")) v = v != sum();
            else if (peek_single('<')) { pos_++; v = v < sum(); }
            else if (peek_single('>')) { pos_++; v = v > sum(); }
            else return v;
        }
    }

    bool eat2(const char* op) {
        skip_ws();
        if (s_.compare(pos_, 2, op) == 0) { pos_ += 2; return true; }
        return false;
    }

    // A one-character operator that is not the start of a two-character one.
    bool peek_single(char c) {
        skip_ws();
        return pos_ < s_.size() && s_[pos_] == c && (pos_ + 1 >= s_.size() || s_[pos_ + 1] != '=');
    }

    int64_t sum() {
        int64_t v = product();
        for (;;) {
            if (eat('+')) v += product();
            else if (eat('-')) v -= product();
            else return v;
        }
    }

    int64_t product() {
        int64_t v = unary();
        for (;;) {
            if (eat('*')) v *= unary();
            else if (eat('/')) v = div_checked(v, unary(), *this);
            else if (eat('%')) {
                const int64_t d = unary();
                if (d == 0) {
                    if (!skip_) fail("modulo by zero");
                    v = 0;
                } else {
                    v %= d;
                }
            }
            else return v;
        }
    }

    int64_t unary() {
        if (eat('-')) return -unary();
        if (pos_ < s_.size() && s_[pos_] == '!' && (pos_ + 1 >= s_.size() || s_[pos_ + 1] != '=')) {
            pos_++;
            return unary() ? 0 : 1;
        }
        if (eat('+')) return unary();
        return primary();
    }

    int64_t primary() {
        skip_ws();
        if (eat('(')) {
            const int64_t v = conditional();
            if (!eat(')')) fail("missing ')'");
            return v;
        }
        if (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
            int64_t v = 0;
            while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) {
                v = v * 10 + (s_[pos_++] - '0');
            }
            return v;
        }
        if (pos_ < s_.size() && (std::isalpha(static_cast<unsigned char>(s_[pos_])) || s_[pos_] == '_')) {
            const size_t start = pos_;
            while (pos_ < s_.size() &&
                   (std::isalnum(static_cast<unsigned char>(s_[pos_])) || s_[pos_] == '_')) pos_++;
            const std::string name = s_.substr(start, pos_ - start);
            if (eat('(')) return call(name);
            auto it = vars_.find(name);
            if (it == vars_.end()) fail("unknown name '" + name + "'");
            return it->second;
        }
        if (pos_ >= s_.size()) fail("unexpected end");
        fail("unexpected '" + std::string(1, s_[pos_]) + "'");
    }

    int64_t call(const std::string& name) {
        std::vector<int64_t> args;
        if (!eat(')')) {
            do { args.push_back(conditional()); } while (eat(','));
            if (!eat(')')) fail("missing ')' after arguments of " + name);
        }
        if (name == "cdiv") {
            if (args.size() != 2) fail("cdiv takes 2 arguments");
            if (args[1] == 0) {
                if (skip_) return 0;
                fail("division by zero");
            }
            return (args[0] + args[1] - 1) / args[1];
        }
        if (name == "min" || name == "max") {
            if (args.empty()) fail(name + " needs at least 1 argument");
            int64_t v = args[0];
            for (auto a : args) v = (name == "min") ? std::min(v, a) : std::max(v, a);
            return v;
        }
        fail("unknown function '" + name + "'");
    }

    const std::string& s_;
    const Vars& vars_;
    size_t pos_ = 0;
    int skip_ = 0;   // > 0 inside an operand whose value is not used
};

}

int64_t eval(const std::string& expr, const Vars& vars) {
    return Parser(expr, vars).parse();
}

int64_t eval_json(const nlohmann::json& v, const Vars& vars) {
    if (v.is_number_integer()) return v.get<int64_t>();
    if (v.is_string()) return eval(v.get<std::string>(), vars);
    throw std::runtime_error("expected an integer or an expression string, got " + v.dump());
}

}
