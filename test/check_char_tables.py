"""One-off check: the character tables derived from OPEN_TO_CLOSE after the refactor must cover the hand-copied literals of the old version.

The content of the old sets comes from chinese_linebreak.py before the refactor (commit 899c316).
Only two kinds of difference are expected: 1) whitespace added to STRUCTURAL_BREAK_CHARS; 2) the derived tables add the brackets the old tables missed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from manga_translator.rendering import chinese_linebreak as cl


OLD_STRUCTURAL = set(
    "，、。．｡､,.!?！？；;：:﹐﹑﹒﹔﹕﹖﹗︐︑︒︓︔︕︖…‥⋯︰⋮︙︴—－–−︱︲～〜〰~≀|"
)
OLD_PHRASE_PUNCT = set(
    "，。！？；：、,.!?;:．｡､﹐﹑﹒﹔﹕﹖﹗︐︑︒︓︔︕︖"
    "…‥⋯︰⋮︙︴～〜〰—－–−︱︲─│━┃═║~≀|·・﹅‚„"
    "()（）[]［］{}｛｝【】〔〕〖〗〘〙〚〛"
    "「」『』｢｣《》〈〉"
    "⁅⁆⟦⟧⟨⟩⟪⟫⦃⦄⦅⦆⦇⦈⦉⦊⦋⦌⦍⦎⦏⦐⦑⦒⧼⧽"
    "︵︶︷︸︹︺︻︼︽︾︿﹀"
    "﹁﹂﹃﹄﹙﹚﹛﹜﹝﹞﹇﹈"
)
OLD_NO_START = set(
    "，、。．｡､,.!?！？；;：:﹐﹑﹒﹔﹕﹖﹗︐︑︒︓︔︕︖"
    "…‥⋯︰⋮︙︴—－–−︱︲～〜〰~≀|·・﹅"
    "”’〞〟＂＇»›"
    "》，」』】）﹂﹄︶︸︺︼︾﹀﹚﹜﹞﹈)]｝｣》〉"
    "⁆⟧⟩⟫⦄⦆⦈⦊⦌⦎⦐⦒⧽"
)
OLD_NO_END = set("《「『【（﹁﹃︵︷︹︻︽︿﹙﹛﹝﹇([{｛｢〈⁅⟦⟨⟪⦃⦅⦇⦉⦋⦍⦏⦑⧼")


def main() -> int:
    ok = True
    for name, old, new in [
        ("STRUCTURAL_BREAK_CHARS", OLD_STRUCTURAL, cl.STRUCTURAL_BREAK_CHARS),
        ("PHRASE_PUNCT", OLD_PHRASE_PUNCT, cl.PHRASE_PUNCT),
        ("NO_START_CHARS", OLD_NO_START, cl.NO_START_CHARS),
        ("NO_END_CHARS", OLD_NO_END, cl.NO_END_CHARS),
    ]:
        missing = old - new
        added = new - old
        print(f"{name}: 缺失={ascii(sorted(missing))} 新增={ascii(sorted(added))}")
        if missing:
            ok = False
    print("OK" if ok else "FAIL: 派生表丢了旧表字符")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
