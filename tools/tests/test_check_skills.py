"""Tests for tools/check_skills.py against throwaway repo layouts."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import check_skills  # noqa: E402

SKILL = """---
name: {name}
description: {desc}
---

See `tools/pgaio.py`.
"""


class TestCheckSkills(unittest.TestCase):
    def make_repo(self, skill_name="demo", desc="Does a thing.", links=True):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        r = Path(tmp.name)
        (r / "skills/demo").mkdir(parents=True)
        (r / "skills/demo/SKILL.md").write_text(SKILL.format(name=skill_name, desc=desc))
        (r / "tools").mkdir()
        (r / "tools/pgaio.py").write_text("")
        if links:
            for d in (".agents", ".claude"):
                (r / d).mkdir()
                os.symlink("../skills", r / d / "skills")
        return r

    def test_valid_layout(self):
        self.assertEqual(check_skills.check(self.make_repo()), [])

    def test_the_real_repo(self):
        self.assertEqual(check_skills.check(), [])

    def test_name_mismatch(self):
        p = check_skills.check(self.make_repo(skill_name="other"))
        self.assertTrue(any("does not match folder" in x for x in p), p)

    def test_long_description(self):
        p = check_skills.check(self.make_repo(desc="x" * 1100))
        self.assertTrue(any("limit 1024" in x for x in p), p)

    def test_missing_links(self):
        p = check_skills.check(self.make_repo(links=False))
        self.assertEqual(len([x for x in p if "symlink" in x]), 2, p)

    def test_missing_referenced_path(self):
        r = self.make_repo()
        os.remove(r / "tools/pgaio.py")
        p = check_skills.check(r)
        self.assertTrue(any("missing path tools/pgaio.py" in x for x in p), p)


if __name__ == "__main__":
    unittest.main()
