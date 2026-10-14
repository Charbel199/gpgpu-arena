#!/usr/bin/env python3
"""Checks the skills in skills/ without needing Claude Code or Codex.

    python3 tools/check_skills.py

Exits non-zero with one line per problem. Run by CI.
"""

import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def frontmatter(text):
    """The YAML frontmatter as a flat dict of top-level `key: value` lines."""
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---", 4)
    if end < 0:
        return None
    out = {}
    for line in text[4:end].splitlines():
        m = re.match(r"^([A-Za-z_-]+):\s*(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def check(repo=REPO):
    problems = []

    skills_dir = repo / "skills"
    skills = sorted(d for d in skills_dir.iterdir() if d.is_dir()) if skills_dir.exists() else []
    if not skills:
        problems.append("skills/: no skills found")
    for d in skills:
        md = d / "SKILL.md"
        rel = md.relative_to(repo)
        if not md.exists():
            problems.append(f"{rel}: missing")
            continue
        text = md.read_text()
        fm = frontmatter(text)
        if fm is None:
            problems.append(f"{rel}: no frontmatter")
            continue
        if fm.get("name") != d.name:
            problems.append(f"{rel}: name '{fm.get('name')}' does not match folder '{d.name}'")
        desc = fm.get("description", "")
        if not desc:
            problems.append(f"{rel}: missing description")
        elif len(desc) > 1024:
            problems.append(f"{rel}: description is {len(desc)} chars, limit 1024")
        if len(text.splitlines()) > 500:
            problems.append(f"{rel}: over 500 lines")
        # Every repo path a skill names in backticks should exist, so docs
        # cannot drift from the tree unnoticed.
        for path in re.findall(r"`((?:kernels|tools|pgaio|include|src|skills)/[^`\s<>*]+)`", text):
            path = path.rstrip("/.,")
            if "{" in path or "<" in path:
                continue
            if not (repo / path).exists():
                problems.append(f"{rel}: refers to missing path {path}")

    for link in (".agents/skills", ".claude/skills"):
        p = repo / link
        if not p.is_symlink() or os.path.realpath(p) != str(skills_dir.resolve()):
            problems.append(f"{link}: should be a symlink to ../skills")
    return problems


def main():
    problems = check()
    for p in problems:
        print(p)
    if problems:
        sys.exit(1)
    print("skills OK")


if __name__ == "__main__":
    main()
