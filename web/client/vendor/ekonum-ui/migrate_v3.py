#!/usr/bin/env python3
"""Rewrites an application's files from ekonum-ui 2.x names to 3.0.0 names.

    migrate_v3.py --check templates/ static/app.css   # lists what would change, writes nothing
    migrate_v3.py templates/ static/app.css           # rewrites in place

Covers classes (.ekn-carte → .ekn-card), custom properties (--ekn-vert-fonce →
--ekn-dark-green), Tailwind utilities from the ekonum-ui theme (bg-ekn-vert-fonce →
bg-ekn-dark-green), data-libelle, and paths inside the archive (ekonum-ui/polices/ →
ekonum-ui/fonts/). The name map is migrations/v3.json, shipped in the archive next to
this script. Review the diff: a French word that merely looks like an old name is only
rewritten when it carries the ekn- prefix, so plain text is never touched.

Standard library only.
"""
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
MAP_CANDIDATES = [HERE / "migrations" / "v3.json", HERE.parent / "migrations" / "v3.json"]
EXTENSIONS = {".html", ".htm", ".jinja", ".jinja2", ".j2", ".njk", ".hbs", ".css", ".scss", ".sass", ".less",
              ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".vue", ".svelte", ".astro", ".py", ".md", ".yml", ".yaml"}
SKIP_DIRS = {".git", "node_modules", "dist", "build", ".venv", "venv", "__pycache__", "ekonum-ui"}
WORD = r"[\w-]"


def load_map():
    for path in MAP_CANDIDATES:
        if path.exists():
            return json.loads(path.read_text())
    sys.exit("✗ migrations/v3.json not found next to this script")


def rules(m):
    """Ordered (pattern, replacement) pairs. Longest names first, so that a prefix never wins."""
    out = []

    def by_length(d):
        return sorted(((k, v) for k, v in d.items() if k != v), key=lambda kv: -len(kv[0]))

    # Custom properties: --ekn-vert-fonce, --ekn-colonne…
    for old, new in by_length({**m["tokens"], **m["properties"]}):
        out.append((re.compile(rf"--ekn-{re.escape(old)}(?!{WORD})"), f"--ekn-{new}"))
    # Classes, as whole words: not preceded by "-" (that would be a property or a utility).
    for old, new in by_length(m["classes"]):
        out.append((re.compile(rf"(?<!{WORD}){re.escape(old)}(?!{WORD})"), new))
    # A block followed by a modifier built at run time: class="ekn-niveau--${state}".
    # The block is renamed here; the modifier's values live in the application's code
    # (main() flags them).
    for old, new in by_length({k: v for k, v in m["classes"].items() if "--" not in k and "__" not in k}):
        out.append((re.compile(rf"(?<!{WORD}){re.escape(old)}(?=(--|__)[$\{{])"), new))
    # Tailwind utilities of the ekonum-ui theme: <utility>-ekn-<name>. Fonts and line heights
    # first, because "titre" and "texte" mean something else there than for colours.
    for prefix, pairs in (("font", {"titre": "heading", "texte": "body"}),
                          ("leading", {"titre": "heading", "texte": "body"})):
        for old, new in pairs.items():
            out.append((re.compile(rf"(?<!{WORD}){prefix}-ekn-{old}(?!{WORD})"), f"{prefix}-ekn-{new}"))
    utility = {k: v for k, v in m["tokens"].items() if not re.match(r"(t|e|r|ombre|police|degrade)-|inter", k)}
    utility.update({"carte": "card", "plaque": "tile", "champ": "field", "pilule": "pill"})
    for old, new in by_length(utility):
        out.append((re.compile(rf"(?<=[a-z0-9]-)ekn-{re.escape(old)}(?=(/\d+)?(?!{WORD}))"), f"ekn-{new}"))
    # Attributes and archive paths.
    for old, new in m["attributes"].items():
        out.append((re.compile(rf"(?<!{WORD}){re.escape(old)}(?!{WORD})"), new))
    files = m["files"]
    out.append((re.compile(r"ekonum-ui/polices/polices\.css"), "ekonum-ui/fonts/fonts.css"))
    out.append((re.compile(r"ekonum-ui/polices/"), "ekonum-ui/fonts/"))
    for old, new in by_length({k: v for k, v in files.items() if k.endswith(".svg")}):
        out.append((re.compile(rf"ekonum-ui/marque/{re.escape(old)}"), f"ekonum-ui/brand/{new}"))
    out.append((re.compile(r"ekonum-ui/marque/"), "ekonum-ui/brand/"))
    return out


def migrate(text, compiled):
    count = 0
    for pattern, replacement in compiled:
        text, n = pattern.subn(replacement, text)
        count += n
    return text, count


def files(paths):
    for p in map(pathlib.Path, paths):
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix in EXTENSIONS and not SKIP_DIRS & set(f.relative_to(p).parts[:-1]):
                    yield f


def main():
    args = sys.argv[1:]
    check = "--check" in args
    paths = [a for a in args if a != "--check"]
    if not paths:
        sys.exit(__doc__)
    compiled = rules(load_map())
    total = changed = 0
    for f in files(paths):
        try:
            before = f.read_text()
        except UnicodeDecodeError:
            continue
        after, n = migrate(before, compiled)
        if n:
            changed += 1
            total += n
            print(f"{'~' if check else '✓'} {f}: {n} name(s)")
            if not check:
                f.write_text(after)
        dynamic = sorted(set(re.findall(r"ekn-[a-z-]+?(?:--|__)[$\{][^\"'` ]*", after)))
        if dynamic:
            print(f"  ! {f}: modifier built at run time, map its values by hand "
                  f"(ok → success, attention → warning, echec → error, en-cours → running…): {', '.join(dynamic)}")
        # Names the map does not know: a typo, or a class that never existed.
        left = sorted(set(re.findall(r"--ekn-(?:vert|blanc|texte|trait|survol|echec|inconnu|police|interl|cible|gouttiere|largeur|degrade|ombre)[\w-]*", after)))
        if left:
            print(f"  ! {f}: still French after migration: {', '.join(left)}")
    verb = "would change" if check else "changed"
    print(f"{'~' if check else '✓'} {total} name(s) {verb} in {changed} file(s)")


if __name__ == "__main__":
    main()
