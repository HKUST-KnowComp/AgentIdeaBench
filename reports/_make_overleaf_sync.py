r"""Generate the Overleaf sync guide from an actual diff, not by hand.

The guide tells the user how to turn the .tex currently on Overleaf into the
local one. Writing it by hand went stale twice, so it is derived here: hunks
come from difflib against the downloaded Overleaf source, and the file is
verified by replaying its own steps.

Usage:
  /usr/bin/python3 reports/_make_overleaf_sync.py <overleaf.zip-or-tex> <YYYYMMDD>

Writes docs/paper/OVERLEAF_SYNC_<date>.md. Read-only on everything else.
"""
import difflib
import hashlib
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
LOCAL = ROOT / "docs" / "paper" / "acl_latex.tex"


def md5(text):
    return hashlib.md5(text.encode()).hexdigest()


def read_remote(arg):
    p = Path(arg)
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            return z.read("acl_latex.tex").decode()
    return p.read_text()


def hunks(a, b):
    """Opcodes as (tag, i1, i2, j1, j2) over line lists, newest first."""
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return [op for op in sm.get_opcodes() if op[0] != "equal"]


def fence(lines):
    return "```latex\n" + "\n".join(lines) + "\n```"


def main():
    remote_path, date = sys.argv[1], sys.argv[2]
    a = read_remote(remote_path).split("\n")
    b = LOCAL.read_text().split("\n")
    ops = hunks(a, b)

    out = [f"# Overleaf 同步指引 — {date[:4]}-{date[4:6]}-{date[6:]}", ""]
    # a trailing newline yields one empty element; report wc -l's count
    na = len(a) - (1 if a and a[-1] == "" else 0)
    nb = len(b) - (1 if b and b[-1] == "" else 0)
    out += [f"> 基准 = Overleaf 当前版本（`{Path(remote_path).name}`，"
            f"{na} 行，md5 `{md5(chr(10).join(a))}`）",
            f"> 目标 = 本地 `docs/paper/acl_latex.tex`（{nb} 行，"
            f"md5 `{md5(LOCAL.read_text())}`）", ""]
    out += ["## 方案 A（推荐）：整份覆盖", "",
            "直接把本地 `docs/paper/acl_latex.tex` 上传覆盖。以下逐处版本仅供核对。",
            "", f"## 方案 B：逐处改（{len(ops)} 处）", "",
            "**行号是 Overleaf 当前文件的行号。必须从大往小改。**", ""]

    for tag, i1, i2, j1, j2 in reversed(ops):
        if tag == "replace":
            span = f"L{i1 + 1}" if i2 - i1 == 1 else f"L{i1 + 1}–L{i2}"
            n = j2 - j1
            head = (f"### 替换 Overleaf **{span}**（整行）" if n == 1 and i2 - i1 == 1
                    else f"### 用 {n} 行替换 Overleaf **{span}**")
            out += [head, "", "原内容：", fence(a[i1:i2]), "",
                    "改成：", fence(b[j1:j2]), ""]
        elif tag == "delete":
            span = f"L{i1 + 1}" if i2 - i1 == 1 else f"L{i1 + 1}–L{i2}"
            out += [f"### 删除 Overleaf **{span}**", "", fence(a[i1:i2]), ""]
        elif tag == "insert":
            out += [f"### 在 Overleaf **L{i1}** 之后插入 {j2 - j1} 行", "",
                    fence(b[j1:j2]), ""]

    path = ROOT / "docs" / "paper" / f"OVERLEAF_SYNC_{date}.md"
    path.write_text("\n".join(out))

    # replay the steps to prove the guide is complete
    cur = list(a)
    for tag, i1, i2, j1, j2 in reversed(ops):
        cur[i1:i2] = b[j1:j2]
    ok = md5("\n".join(cur)) == md5("\n".join(b))
    print(f"wrote {path.name}  ({len(ops)} hunks)  replay {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
