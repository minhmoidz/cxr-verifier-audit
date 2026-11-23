"""Fill plain-text submission files (highlights, cover letter) from the generated number macros.

Templates use {{key}}, {{key@1}} or {{key@0}}, the same keys and formats as \\val{} in the manuscript.
Values come from outputs/generated/numbers.tex and numbers_v2.tex; no number is typed by hand.
Usage: python3 tools/v2_fill_text.py <template> <output> [--max-line-chars N]
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "outputs", "generated")


def load():
    vals = {}
    for f in ("numbers.tex", "numbers_v2.tex"):
        for m in re.finditer(r"\\csname n@(.+?)\\endcsname\{(.*)\}$", open(os.path.join(GEN, f)).read(), re.M):
            vals[m.group(1)] = m.group(2).replace("{,}", ",").replace("$-$", "-")
    return vals


def main():
    tmpl, out = sys.argv[1], sys.argv[2]
    vals = load()
    missing = []

    def sub(m):
        k = m.group(1)
        if k not in vals:
            missing.append(k)
            return "TODO_RESULT_NEEDED:" + k
        return vals[k]

    text = re.sub(r"\{\{([^}]+)\}\}", sub, open(tmpl).read())
    if missing:
        sys.exit(f"missing keys: {missing}")
    if "--max-line-chars" in sys.argv:
        n = int(sys.argv[sys.argv.index("--max-line-chars") + 1])
        for line in text.splitlines():
            if line.startswith("- ") and len(line) - 2 > n:
                sys.exit(f"line longer than {n} characters ({len(line) - 2}): {line}")
    open(out, "w").write(text)
    print("wrote", out)


if __name__ == "__main__":
    main()
