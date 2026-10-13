r"""Lightweight structural validator for the LocalStrength LaTeX sources.

Checks, without a TeX engine:
  - balanced braces { }
  - balanced \begin{env}..\end{env}
  - every \ref/\eqref target has a matching \label
  - every \cite key appears in refs.bib
  - every \includegraphics file exists on disk
  - no leftover ALL_CAPS_PLACEHOLDER tokens

Usage: python cn_tex_check.py <paper_dir>
"""
import re
import sys
from pathlib import Path


def check_file(tex: Path, bibkeys, figdir: Path):
    src = tex.read_text(encoding="utf-8", errors="replace")
    problems = []

    # strip comments (keep escaped \%)
    nocom = re.sub(r"(?<!\\)%.*", "", src)

    # braces
    bal = 0
    for ch in nocom:
        if ch == "{":
            bal += 1
        elif ch == "}":
            bal -= 1
        if bal < 0:
            problems.append("unbalanced '}' (closes before open)")
            break
    if bal != 0:
        problems.append(f"brace imbalance: net {bal}")

    # environments
    begins = re.findall(r"\\begin\{([^}]+)\}", nocom)
    ends = re.findall(r"\\end\{([^}]+)\}", nocom)
    from collections import Counter
    cb, ce = Counter(begins), Counter(ends)
    for env in set(begins) | set(ends):
        if cb[env] != ce[env]:
            problems.append(f"env '{env}': {cb[env]} begin vs {ce[env]} end")

    # labels and refs
    labels = set(re.findall(r"\\label\{([^}]+)\}", nocom))
    refs = re.findall(r"\\(?:eq)?ref\{([^}]+)\}", nocom)
    for r in refs:
        for key in r.split(","):
            key = key.strip()
            if key and key not in labels:
                problems.append(f"ref to missing label: {key}")

    # citations
    cites = re.findall(r"\\cite[a-z]*\{([^}]+)\}", nocom)
    for c in cites:
        for key in c.split(","):
            key = key.strip()
            if key and key not in bibkeys:
                problems.append(f"cite to missing bibkey: {key}")

    # graphics
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", nocom)
    for f in figs:
        fp = (figdir / f)
        cands = [fp, fp.with_suffix(".png"), fp.with_suffix(".pdf")]
        if not any(c.exists() for c in cands):
            problems.append(f"missing figure file: {f}")

    # leftover placeholders
    for m in re.findall(r"\b[A-Z]{3,}_[A-Z0-9_]{2,}\b", nocom):
        if m not in ("JEL",):
            problems.append(f"leftover placeholder: {m}")

    return problems, len(labels), len(set(cites))


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    bib = d / "refs.bib"
    bibkeys = set()
    if bib.exists():
        bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib.read_text(
            encoding="utf-8", errors="replace")))
    ok = True
    for name in ("LocalStrength.tex", "LocalStrength_IA.tex"):
        tex = d / name
        if not tex.exists():
            print(f"[skip] {name} (not found)")
            continue
        probs, nlab, ncite = check_file(tex, bibkeys, d)
        print(f"\n=== {name}: {nlab} labels, {ncite} cite groups, "
              f"{len(bibkeys)} bibkeys ===")
        if probs:
            ok = False
            for p in probs[:60]:
                print("  [X]", p)
            if len(probs) > 60:
                print(f"  ... and {len(probs)-60} more")
        else:
            print("  [OK] no structural problems")
    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
