# /// script
# dependencies = [
#   "rope>=1.14,<1.15",
# ]
# ///

"""A small CLI wrapper around rope for mechanical refactors.

Supported operations:
- move-module: move a Python module (file) into another package.
- rename-module: rename a Python module (file) within its package.
- rename-symbol: rename a symbol based on an offset or regex match.
- extract-function: extract a range of code into a new top-level function.
- extract-method: extract a range of code into a new method.
- inline-variable: inline a variable at the given offset.
- inline-method: inline a method/function call at the given offset.
- organize-imports: remove unused imports and normalize imports (best-effort; rope-version dependent).
- batch: run a sequence of the above operations from a JSON file.

This script is intentionally conservative:
- defaults to dry-run (prints what would change)
- requires explicit --apply to write changes

Run from anywhere; pass --project-root to point at the repo root.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rope.refactor.extract as rope_extract
import rope.refactor.inline as rope_inline
import rope.refactor.move as rope_move
import rope.refactor.rename as rope_rename
from rope.base import libutils, pynames
from rope.base.change import ChangeContents, ChangeSet, MoveResource
from rope.base.fscommands import FileSystemCommands
from rope.base.project import Project
from rope.refactor import importutils


@dataclass(frozen=True)
class Mode:
    apply: bool


def _die(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def _is_repo_root_marker(dir_path: Path) -> bool:
    return (dir_path / ".git").exists()


def _is_python_project_marker(dir_path: Path) -> bool:
    return any((dir_path / f).exists() for f in ["pyproject.toml", "setup.cfg", "setup.py"])


def _auto_project_root(target_path: Path, *, fallback_root: Path) -> Path:
    """Pick a good rope project root for a target file.

    Strategy: walk upward from the target file's directory looking for common
    Python project markers (pyproject/setup.cfg/setup.py). If none are found,
    fall back to the nearest repo root marker (.git) or fallback_root.
    """

    cur = target_path
    if cur.is_file():
        cur = cur.parent

    best_marker: Path | None = None
    while True:
        if _is_python_project_marker(cur):
            best_marker = cur
            break

        if _is_repo_root_marker(cur):
            if best_marker is None:
                best_marker = cur
            break

        if cur.parent == cur:
            break
        cur = cur.parent

    return best_marker or fallback_root


def _paths_to_scan_roots(project_root: Path, paths: list[Path]) -> list[str]:
    """Convert hit file paths to a small set of scan-roots.

    Prefer package-ish directories (those containing __init__.py). For each hit
    file, walk up until a package directory is found, then use that as a scan
    root. Deduplicate and remove redundant subpaths.
    """

    roots: set[str] = set()

    for p in paths:
        try:
            rel = p.resolve().relative_to(project_root.resolve())
        except Exception:
            continue

        cur = project_root / rel
        if cur.is_file():
            cur = cur.parent

        pkg_dir: Path | None = None
        while True:
            if (cur / "__init__.py").exists():
                pkg_dir = cur
                break
            if cur == project_root or cur.parent == cur:
                break
            cur = cur.parent

        if pkg_dir is None:
            pkg_dir = project_root / rel.parts[0] if rel.parts else project_root

        roots.add(str(pkg_dir.relative_to(project_root)))

    root_list = sorted(roots, key=lambda s: (s.count("/"), len(s)))
    pruned: list[str] = []
    for r in root_list:
        if any(r == x or r.startswith(x.rstrip("/") + "/") for x in pruned):
            continue
        pruned.append(r)

    return pruned or ["."]


def _rg_list_files(project_root: Path, *, needle: str) -> list[Path]:
    """Return files containing `needle` using ripgrep if available."""

    try:
        proc = subprocess.run(
            ["rg", "-l", "-F", needle, "-g", "*.py", "."],
            cwd=str(project_root),
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []

    if proc.returncode not in (0, 1):
        return []

    hits: list[Path] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        hits.append((project_root / line).resolve())
    return hits


def _python_list_files(project_root: Path, *, needle: str) -> list[Path]:
    """Fallback reference finder without rg (slower)."""

    hits: list[Path] = []
    for p in project_root.rglob("*.py"):
        try:
            s = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if needle in s:
            hits.append(p.resolve())
    return hits


def _auto_scan_roots(project_root: Path, *, needles: list[str], include_paths: list[Path]) -> list[str]:
    """Pick scan-roots by locating references for one or more needles."""

    hit_paths: list[Path] = []
    for n in needles:
        hit_paths.extend(_rg_list_files(project_root, needle=n))

    if not hit_paths:
        for n in needles[:1]:
            hit_paths.extend(_python_list_files(project_root, needle=n))

    hit_paths.extend([p.resolve() for p in include_paths if p.exists()])

    return _paths_to_scan_roots(project_root, hit_paths)


def _python_file_resources(project_root: Path, roots: list[str]) -> list[str]:
    """Expand scan roots to a list of python file relpaths."""

    files: list[str] = []
    for r in roots:
        base = (project_root / r).resolve()
        if not base.exists():
            continue
        if base.is_file() and base.suffix == ".py":
            try:
                files.append(str(base.relative_to(project_root)))
            except Exception:
                pass
            continue
        for p in sorted(base.rglob("*.py")):
            try:
                files.append(str(p.relative_to(project_root)))
            except Exception:
                continue

    seen: set[str] = set()
    out: list[str] = []
    for f in files:
        if f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out


def _file_to_dotted_module_name(project_root: Path, rel_file: str) -> str | None:
    """Best-effort conversion of a .py path (relative to project_root) to a dotted module name."""

    try:
        p = (project_root / rel_file).resolve()
        rel = p.relative_to(project_root.resolve())
    except Exception:
        return None

    if p.suffix != ".py":
        return None

    parts = list(rel.with_suffix("").parts)
    if not parts:
        return None

    pkg_start = 0
    for i in range(len(parts)):
        d = project_root / Path(*parts[: i + 1])
        if d.is_dir() and (d / "__init__.py").exists():
            pkg_start = i
            break

    return ".".join(parts[pkg_start:])


def _open_project(project_root: Path) -> Project:
    """Open a rope Project using plain filesystem ops.

    Rope will try to use VCS-aware moves (e.g. `git mv`) when it detects a VCS.
    That can fail for untracked files; using FileSystemCommands makes refactors
    usable in scratch dirs and partially-tracked repos.
    """

    return Project(str(project_root), fscommands=FileSystemCommands())


def _print_changes(changes) -> None:
    # Rope changes objects vary, but most have these.
    desc = getattr(changes, "get_description", None)
    if callable(desc):
        print(desc())
    else:
        print("(no description available)")

    changed = getattr(changes, "get_changed_resources", None)
    if callable(changed):
        resources = list(changed())
        if resources:
            print("Files to be modified:")
            for r in resources:
                p = getattr(r, "path", None)
                print(f"- {p if p is not None else r}")


def _do_or_preview(project, changes, mode: Mode) -> None:
    _print_changes(changes)
    if mode.apply:
        project.do(changes)
        print("Applied.")
    else:
        print("Dry-run only (pass --apply to write changes).")


def _resource_text(project, relpath: str) -> tuple[Any, str]:
    res = project.get_resource(relpath)
    return res, res.read()


def _nth_match(pattern: str, text: str, occurrence: int) -> re.Match[str] | None:
    if occurrence < 1:
        _die("--occurrence must be >= 1")
    it = re.finditer(pattern, text, flags=re.MULTILINE)
    for i, m in enumerate(it, start=1):
        if i == occurrence:
            return m
    return None


def _compute_offset(
    *,
    text: str,
    offset: int | None,
    pattern: str | None,
    occurrence: int,
    group: int,
    symbol: str | None,
    context: str,
) -> int:
    if offset is not None:
        if offset < 0 or offset >= len(text):
            _die(f"--offset out of bounds (0..{len(text) - 1})")
        return offset

    # If caller provides --symbol without --pattern, build a safer pattern that
    # targets the definition site and uses a capture group around the symbol.
    if symbol is not None and pattern is None:
        sym = re.escape(symbol)
        if context == "class":
            pattern = rf"\bclass\s+({sym})\b"
            group = 1
        elif context == "def":
            pattern = rf"\bdef\s+({sym})\b"
            group = 1
        elif context == "any":
            pattern = rf"\b({sym})\b"
            group = 1
        else:
            _die("--context must be one of: class, def, any")

    if not pattern:
        _die("Provide either --offset, --pattern, or --symbol")

    m = _nth_match(pattern, text, occurrence)
    if not m:
        _die(f"Pattern not found (occurrence {occurrence}): {pattern!r}")

    try:
        return m.start(group)
    except IndexError:
        _die(f"Match group {group} does not exist for pattern: {pattern!r}")


def _line_range_to_offsets(text: str, start_line: int, end_line: int) -> tuple[int, int]:
    if start_line < 1 or end_line < 1:
        _die("--start-line/--end-line must be >= 1")
    if end_line < start_line:
        _die("--end-line must be >= --start-line")

    lines = text.splitlines(True)
    if start_line > len(lines):
        _die(f"--start-line out of range (file has {len(lines)} lines)")
    if end_line > len(lines):
        _die(f"--end-line out of range (file has {len(lines)} lines)")

    start_off = sum(len(l) for l in lines[: start_line - 1])
    end_off = sum(len(l) for l in lines[:end_line])
    return start_off, end_off


def _move_module(
    *,
    project_root: Path,
    src: str,
    dest_package: str,
    scan_roots: list[str] | None,
    mode: Mode,
) -> None:
    """Move a module file into another *package* and update imports/usages."""

    project = _open_project(project_root)

    try:
        src_res, _src_text = _resource_text(project, src)

        if not hasattr(rope_move, "MoveModule"):
            _die("Your rope version does not expose MoveModule; try upgrading rope.")

        # Rope expects a destination folder Resource that is a Python package.
        dest_relpath = dest_package.replace(".", "/")
        try:
            dest_res = project.get_resource(dest_relpath)
        except Exception:
            _die(f"Destination package folder not found: {dest_relpath} (create it and add __init__.py)")

        if not dest_res.is_folder() or not dest_res.has_child("__init__.py"):
            _die(f"Destination must be a Python package folder with __init__.py: {dest_relpath}")

        mover = rope_move.MoveModule(project, src_res)
        changes = mover.get_changes(dest_res)
        _do_or_preview(project, changes, mode)
    finally:
        project.close()


def _rename_module(
    *,
    project_root: Path,
    src: str,
    new_name: str,
    scan_roots: list[str] | None,
    mode: Mode,
) -> None:
    """Rename a module file (e.g. pkg/a.py -> pkg/alpha.py) and update imports/usages.

    Rope does not expose a stable public "RenameModule" API across versions.
    This implementation uses rope's import analysis and occurrence finding.
    """

    project = _open_project(project_root)

    try:
        src_res, _src_text = _resource_text(project, src)
        if src_res.is_folder() or not src_res.path.endswith(".py"):
            _die("rename-module expects a .py file path")

        old_basename = src_res.name[:-3]

        parent_pkg = src_res.parent
        pkg_modname = libutils.modname(parent_pkg)
        old_fullname = f"{pkg_modname}.{old_basename}" if pkg_modname else old_basename
        new_fullname = f"{pkg_modname}.{new_name}" if pkg_modname else new_name

        dummy_pymodule = libutils.get_string_module(project, "")
        old_pyname = pynames.ImportedModule(dummy_pymodule, resource=src_res)
        tools = rope_move._MoveTools(project, src_res, old_pyname, old_basename)

        if scan_roots:
            rels = _python_file_resources(project_root, scan_roots)
            resources = [project.get_resource(r) for r in rels]
        else:
            resources = project.get_python_files()
        changes = ChangeSet(f"Renaming module <{old_fullname}> to <{new_fullname}>")

        for module in resources:
            if module == src_res:
                continue

            pymodule = project.get_pymodule(module)
            if not tools.occurs_in_module(pymodule=pymodule, resource=module):
                continue

            module_imports = importutils.get_module_imports(project, pymodule)
            changed_imports = False

            for import_stmt in module_imports.imports:
                info = import_stmt.import_info

                # `import pkg.a`
                if isinstance(info, importutils.NormalImport):
                    new_nas = []
                    for name, alias in info.names_and_aliases:
                        if name == old_fullname:
                            new_nas.append((new_fullname, alias))
                            changed_imports = True
                        else:
                            new_nas.append((name, alias))
                    info.names_and_aliases = new_nas

                # `from pkg.a import X` or `from pkg import a`
                elif isinstance(info, importutils.FromImport):
                    if info.module_name == old_fullname:
                        info.module_name = new_fullname
                        changed_imports = True
                    elif pkg_modname and info.module_name == pkg_modname:
                        new_nas = []
                        for name, alias in info.names_and_aliases:
                            if name == old_basename:
                                new_nas.append((new_name, alias))
                                changed_imports = True
                            else:
                                new_nas.append((name, alias))
                        info.names_and_aliases = new_nas

            if changed_imports:
                source = module_imports.get_changed_source()
                source = tools.new_source(pymodule, source)
                pymodule = tools.new_pymodule(pymodule, source)
            else:
                source = None

            # Update occurrences (e.g. `pkg.a`, `from pkg import a` usages) using rope's occurrence finder.
            source = tools.rename_in_module(
                new_fullname,
                imports=True,
                pymodule=pymodule,
                resource=None if changed_imports else module,
            )
            if source is None:
                continue

            pymodule = tools.new_pymodule(pymodule, source)
            source = tools.new_source(pymodule, source)
            if source != module.read():
                changes.add_change(ChangeContents(module, source))

        new_path = str(Path(src).with_name(new_name + ".py"))
        changes.add_change(MoveResource(src_res, new_path, exact=True))

        _do_or_preview(project, changes, mode)
    finally:
        project.close()


def _rename_symbol(
    *,
    project_root: Path,
    file: str,
    new_name: str,
    offset: int | None,
    pattern: str | None,
    occurrence: int,
    group: int,
    symbol: str | None,
    context: str,
    scan_roots: list[str] | None,
    mode: Mode,
) -> None:
    project = _open_project(project_root)

    try:
        res, text = _resource_text(project, file)
        off = _compute_offset(
            text=text,
            offset=offset,
            pattern=pattern,
            occurrence=occurrence,
            group=group,
            symbol=symbol,
            context=context,
        )

        renamer = rope_rename.Rename(project, res, off)
        # Limit scanning to a subset of files when provided for performance.
        resources = None
        if scan_roots:
            rels = _python_file_resources(project_root, scan_roots)
            resources = [project.get_resource(r) for r in rels]
        try:
            changes = renamer.get_changes(new_name, resources=resources)
        except TypeError:
            changes = renamer.get_changes(new_name)
        _do_or_preview(project, changes, mode)
    finally:
        project.close()


def _extract(
    *,
    project_root: Path,
    file: str,
    new_name: str,
    start_line: int,
    end_line: int,
    kind: str,
    mode: Mode,
) -> None:
    project = _open_project(project_root)

    try:
        res, text = _resource_text(project, file)
        start_off, end_off = _line_range_to_offsets(text, start_line, end_line)

        # Rope versions differ; try common class names.
        extractor_cls = None
        if kind == "function":
            for name in ["ExtractFunction", "ExtractMethod", "Extract"]:
                if hasattr(rope_extract, name):
                    extractor_cls = getattr(rope_extract, name)
                    break
        elif kind == "method":
            for name in ["ExtractMethod", "ExtractFunction", "Extract"]:
                if hasattr(rope_extract, name):
                    extractor_cls = getattr(rope_extract, name)
                    break
        else:
            _die("kind must be 'function' or 'method'")

        if extractor_cls is None:
            _die("Could not find an extract refactoring class in rope.refactor.extract; try upgrading rope.")

        try:
            extractor = extractor_cls(project, res, start_off, end_off)
        except TypeError:
            # Some versions want (project, resource, start, end, ...). Show signature hint.
            _die(
                f"Extract API mismatch for {extractor_cls.__name__}. This rope version likely differs; try upgrading rope."
            )

        get_changes = getattr(extractor, "get_changes", None)
        if not callable(get_changes):
            _die(f"{extractor_cls.__name__} has no get_changes(); try upgrading rope")

        changes = get_changes(new_name)
        _do_or_preview(project, changes, mode)
    finally:
        project.close()


def _inline(
    *,
    project_root: Path,
    file: str,
    offset: int | None,
    pattern: str | None,
    occurrence: int,
    group: int,
    symbol: str | None,
    context: str,
    kind: str,
    mode: Mode,
) -> None:
    project = _open_project(project_root)

    try:
        res, text = _resource_text(project, file)
        off = _compute_offset(
            text=text,
            offset=offset,
            pattern=pattern,
            occurrence=occurrence,
            group=group,
            symbol=symbol,
            context=context,
        )

        inliner_cls = None
        if kind == "variable":
            for name in ["InlineVariable", "Inline"]:
                if hasattr(rope_inline, name):
                    inliner_cls = getattr(rope_inline, name)
                    break
        elif kind == "method":
            for name in ["InlineMethod", "InlineFunction", "Inline"]:
                if hasattr(rope_inline, name):
                    inliner_cls = getattr(rope_inline, name)
                    break
        else:
            _die("kind must be 'variable' or 'method'")

        if inliner_cls is None:
            _die("Could not find an inline refactoring class in rope.refactor.inline; try upgrading rope.")

        try:
            inliner = inliner_cls(project, res, off)
        except TypeError:
            _die(
                f"Inline API mismatch for {inliner_cls.__name__}. This rope version likely differs; try upgrading rope."
            )

        get_changes = getattr(inliner, "get_changes", None)
        if not callable(get_changes):
            _die(f"{inliner_cls.__name__} has no get_changes(); try upgrading rope")

        changes = get_changes()
        _do_or_preview(project, changes, mode)
    finally:
        project.close()


def _organize_imports(
    *,
    project_root: Path,
    files: list[str],
    mode: Mode,
) -> None:
    project = _open_project(project_root)

    try:
        organizer = importutils.ImportOrganizer(project)

        for relpath in files:
            res = project.get_resource(relpath)

            result = organizer.organize_imports(res)
            # Rope versions differ: result can be None (no changes), a Change, or a source string.
            if result is None:
                print(f"Organize imports: {relpath} (no changes)")
                continue
            if hasattr(result, "get_changed_resources"):
                print(f"Organize imports: {relpath}")
                _do_or_preview(project, result, mode)
            elif isinstance(result, str):
                print(f"Organize imports: {relpath}")
                if mode.apply:
                    res.write(result)
                    print("Applied.")
                else:
                    print("Dry-run only (would rewrite file contents).")
            else:
                _die(f"Unexpected ImportOrganizer.organize_imports() return type: {type(result)}")
    finally:
        project.close()


def _load_batch_ops(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        ops = data
    elif isinstance(data, dict) and isinstance(data.get("ops"), list):
        ops = data["ops"]
    else:
        _die("Batch JSON must be a list of ops or an object with an 'ops' list")

    out: list[dict[str, Any]] = []
    for i, op in enumerate(ops, start=1):
        if not isinstance(op, dict):
            _die(f"Op #{i} must be an object")
        if "op" not in op:
            _die(f"Op #{i} missing required key: 'op'")
        out.append(op)
    return out


def _collect_py_files(project_root: Path, roots: Iterable[str]) -> list[str]:
    rels: list[str] = []
    for r in roots:
        base = (project_root / r).resolve()
        if not base.exists():
            _die(f"Path does not exist: {r}")
        if base.is_file():
            if base.suffix == ".py":
                rels.append(str(base.relative_to(project_root)))
            continue
        for p in sorted(base.rglob("*.py")):
            rels.append(str(p.relative_to(project_root)))
    return rels


def _run_batch(*, project_root: Path, map_file: Path, mode: Mode) -> None:
    ops = _load_batch_ops(map_file)
    for i, op in enumerate(ops, start=1):
        op_name = op.get("op")
        print(f"== Op {i}/{len(ops)}: {op_name} ==")

        if op_name == "move-module":
            _move_module(
                project_root=project_root,
                src=op["src"],
                dest_package=op["dest_package"],
                scan_roots=None,
                mode=mode,
            )
        elif op_name == "rename-module":
            _rename_module(
                project_root=project_root,
                src=op["src"],
                new_name=op["new_name"],
                scan_roots=None,
                mode=mode,
            )
        elif op_name == "rename-symbol":
            _rename_symbol(
                project_root=project_root,
                file=op["file"],
                new_name=op["new_name"],
                offset=op.get("offset"),
                pattern=op.get("pattern"),
                occurrence=int(op.get("occurrence", 1)),
                group=int(op.get("group", 0)),
                symbol=op.get("symbol"),
                context=str(op.get("context", "any")),
                scan_roots=None,
                mode=mode,
            )
        elif op_name == "extract-function":
            _extract(
                project_root=project_root,
                file=op["file"],
                new_name=op["new_name"],
                start_line=int(op["start_line"]),
                end_line=int(op["end_line"]),
                kind="function",
                mode=mode,
            )
        elif op_name == "extract-method":
            _extract(
                project_root=project_root,
                file=op["file"],
                new_name=op["new_name"],
                start_line=int(op["start_line"]),
                end_line=int(op["end_line"]),
                kind="method",
                mode=mode,
            )
        elif op_name == "inline-variable":
            _inline(
                project_root=project_root,
                file=op["file"],
                offset=op.get("offset"),
                pattern=op.get("pattern"),
                occurrence=int(op.get("occurrence", 1)),
                group=int(op.get("group", 0)),
                symbol=op.get("symbol"),
                context=str(op.get("context", "any")),
                kind="variable",
                mode=mode,
            )
        elif op_name == "inline-method":
            _inline(
                project_root=project_root,
                file=op["file"],
                offset=op.get("offset"),
                pattern=op.get("pattern"),
                occurrence=int(op.get("occurrence", 1)),
                group=int(op.get("group", 0)),
                symbol=op.get("symbol"),
                context=str(op.get("context", "any")),
                kind="method",
                mode=mode,
            )
        elif op_name == "organize-imports":
            # Accept either explicit files or roots for scanning.
            files = op.get("files")
            roots = op.get("roots")
            if isinstance(files, list):
                file_list = [str(x) for x in files]
            elif isinstance(roots, list):
                file_list = _collect_py_files(project_root, [str(x) for x in roots])
            else:
                _die("organize-imports batch op requires 'files' or 'roots'")
            _organize_imports(project_root=project_root, files=file_list, mode=mode)
        else:
            _die(f"Unknown batch op: {op_name}")


def _self_test() -> None:
    """Smoke-test the wrapper against a temporary toy project.

    This validates that the wrapper works end-to-end with the currently installed
    rope, without requiring a real repo.
    """

    with tempfile.TemporaryDirectory(prefix="rope_refactor_self_test_") as td:
        root = Path(td) / "proj"
        pkg = root / "pkg"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("", encoding="utf-8")

        (pkg / "a.py").write_text(
            """
class OldName:
    def __init__(self, x: int) -> None:
        self.x = x

    def double(self) -> int:
        return self.x * 2


def make() -> OldName:
    temp = OldName(21)
    return temp
""".lstrip(),
            encoding="utf-8",
        )

        (pkg / "b.py").write_text(
            """
from pkg.a import OldName, make


def use() -> int:
    obj = OldName(10)
    x = obj.double()
    y = make().double()
    return x + y
""".lstrip(),
            encoding="utf-8",
        )

        # Independent file for extract/inline tests.
        (pkg / "extras.py").write_text(
            """
from __future__ import annotations


def f(n: int) -> int:
    x = n + 1
    y = n + 2
    z = x + y
    return z
""".lstrip(),
            encoding="utf-8",
        )

        # 1) rename-symbol (apply)
        _rename_symbol(
            project_root=root,
            file="pkg/a.py",
            new_name="NewName",
            offset=None,
            pattern=None,
            occurrence=1,
            group=0,
            symbol="OldName",
            context="class",
            scan_roots=None,
            mode=Mode(apply=True),
        )
        assert "class NewName" in (pkg / "a.py").read_text(encoding="utf-8")
        assert "from pkg.a import NewName" in (pkg / "b.py").read_text(encoding="utf-8")

        # 2) rename-module (apply)
        _rename_module(
            project_root=root,
            src="pkg/a.py",
            new_name="alpha",
            scan_roots=None,
            mode=Mode(apply=True),
        )
        assert (pkg / "alpha.py").exists()
        assert "from pkg.alpha import NewName" in (pkg / "b.py").read_text(encoding="utf-8")

        # 3) move-module (apply)
        subpkg = pkg / "subpkg"
        subpkg.mkdir(parents=True)
        (subpkg / "__init__.py").write_text("", encoding="utf-8")
        _move_module(
            project_root=root,
            src="pkg/alpha.py",
            dest_package="pkg.subpkg",
            scan_roots=None,
            mode=Mode(apply=True),
        )
        assert (subpkg / "alpha.py").exists()
        assert "from pkg.subpkg.alpha import NewName" in (pkg / "b.py").read_text(encoding="utf-8")

        # 4) organize-imports (dry-run; should not crash)
        _organize_imports(project_root=root, files=["pkg/b.py"], mode=Mode(apply=False))

        # 5) inline-variable (dry-run; should not crash)
        _inline(
            project_root=root,
            file="pkg/extras.py",
            offset=None,
            pattern=r"\bx\s*=",
            occurrence=1,
            group=0,
            symbol=None,
            context="any",
            kind="variable",
            mode=Mode(apply=False),
        )

        # 6) extract-function (dry-run; should not crash)
        _extract(
            project_root=root,
            file="pkg/extras.py",
            new_name="g",
            start_line=6,
            end_line=7,
            kind="function",
            mode=Mode(apply=False),
        )
        # 7) batch (dry-run)
        ops_path = Path(td) / "ops.json"
        ops_path.write_text(
            """[
  {"op": "rename-symbol", "file": "pkg/subpkg/alpha.py", "symbol": "NewName", "context": "class", "new_name": "RenamedAgain"},
  {"op": "organize-imports", "files": ["pkg/b.py"]}
]
""",
            encoding="utf-8",
        )
        _run_batch(project_root=root, map_file=ops_path, mode=Mode(apply=False))

    print("Self-test passed.")


def _normalize_global_args(argv: list[str]) -> list[str]:
    """Allow global flags (e.g. --project-root/--apply) to appear after subcommands.

    argparse subparsers normally require global flags to appear before the subcommand.
    This reorders known global flags to the front so the skill's examples work.
    """

    reordered: list[str] = []
    rest: list[str] = []

    i = 0
    while i < len(argv):
        tok = argv[i]

        if tok in ("--dry-run", "--apply"):
            reordered.append(tok)
            i += 1
            continue

        if tok in ("--auto-project-root", "--auto-scan-roots"):
            reordered.append(tok)
            i += 1
            continue

        if tok == "--scan-roots":
            reordered.append(tok)
            i += 1
            # Consume values until the next flag.
            while i < len(argv) and not argv[i].startswith("-"):
                reordered.append(argv[i])
                i += 1
            continue

        if tok == "--project-root":
            if i + 1 >= len(argv):
                _die("--project-root requires a value")
            reordered.extend([tok, argv[i + 1]])
            i += 2
            continue

        if tok.startswith("--project-root="):
            reordered.append(tok)
            i += 1
            continue

        rest.append(tok)
        i += 1

    return reordered + rest


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="rope_refactor.py")
    ap.add_argument(
        "--project-root",
        default=".",
        help="Path to the Python project root (where you'd run mypy/pytest).",
    )

    ap.add_argument(
        "--auto-project-root",
        action="store_true",
        help="Auto-detect a good project root based on the target file path.",
    )
    ap.add_argument(
        "--scan-roots",
        nargs="*",
        default=None,
        help="Limit scanning to these subdirectories/files (relative to project-root).",
    )
    ap.add_argument(
        "--auto-scan-roots",
        action="store_true",
        help="Auto-pick scan roots by searching for references (uses rg when available).",
    )

    mx = ap.add_mutually_exclusive_group()
    mx.add_argument("--dry-run", action="store_true", help="Print changes only (default).")
    mx.add_argument("--apply", action="store_true", help="Apply changes to disk.")

    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("version", help="Print rope version")

    p_move = sp.add_parser("move-module", help="Move a module into another package")
    p_move.add_argument(
        "--src",
        required=True,
        help="Path to module .py file, relative to project-root",
    )
    p_move.add_argument(
        "--dest-package",
        required=True,
        help="Destination dotted package path (e.g. pkg.subpkg)",
    )

    p_rmod = sp.add_parser("rename-module", help="Rename a module file within its package")
    p_rmod.add_argument(
        "--src",
        required=True,
        help="Path to module .py file, relative to project-root",
    )
    p_rmod.add_argument(
        "--new-name",
        required=True,
        help="New module name without .py (e.g. 'spec_builder')",
    )

    def _add_location_args(p) -> None:
        p.add_argument("--file", required=True, help="Python file path relative to project-root")
        p.add_argument(
            "--offset",
            type=int,
            default=None,
            help="Character offset pointing at the symbol occurrence",
        )
        p.add_argument(
            "--pattern",
            default=None,
            help="Regex to locate the occurrence (used with --occurrence/--group)",
        )
        p.add_argument(
            "--occurrence",
            type=int,
            default=1,
            help="Which regex match to use (1-based) when using --pattern/--symbol",
        )
        p.add_argument(
            "--group",
            type=int,
            default=0,
            help="Which regex group start to use for the offset (default 0)",
        )
        p.add_argument(
            "--symbol",
            default=None,
            help="Symbol name to target; builds a safer pattern when --pattern is omitted",
        )
        p.add_argument(
            "--context",
            choices=["class", "def", "any"],
            default="any",
            help="When using --symbol without --pattern, what kind of definition to target",
        )

    p_rsym = sp.add_parser(
        "rename-symbol",
        help="Rename a symbol (class/function/var) via offset/pattern/symbol",
    )
    _add_location_args(p_rsym)
    p_rsym.add_argument("--new-name", required=True, help="Replacement identifier")

    p_ef = sp.add_parser("extract-function", help="Extract a line range into a new function")
    p_ef.add_argument("--file", required=True, help="Python file path relative to project-root")
    p_ef.add_argument("--start-line", type=int, required=True)
    p_ef.add_argument("--end-line", type=int, required=True)
    p_ef.add_argument("--new-name", required=True, help="New function name")

    p_em = sp.add_parser("extract-method", help="Extract a line range into a new method")
    p_em.add_argument("--file", required=True, help="Python file path relative to project-root")
    p_em.add_argument("--start-line", type=int, required=True)
    p_em.add_argument("--end-line", type=int, required=True)
    p_em.add_argument("--new-name", required=True, help="New method name")

    p_iv = sp.add_parser("inline-variable", help="Inline a variable at the given offset/pattern")
    _add_location_args(p_iv)

    p_im = sp.add_parser(
        "inline-method",
        help="Inline a method/function call at the given offset/pattern",
    )
    _add_location_args(p_im)

    p_org = sp.add_parser("organize-imports", help="Organize imports for one or more files")
    p_org.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Explicit .py files relative to project-root",
    )
    p_org.add_argument(
        "--roots",
        nargs="*",
        default=None,
        help="Directories to scan for .py files (or individual .py files)",
    )

    p_batch = sp.add_parser("batch", help="Run a batch of operations from a JSON file")
    p_batch.add_argument("--map", required=True, help="Path to JSON file containing ops")

    sp.add_parser("self-test", help="Run a small end-to-end smoke test")

    args = ap.parse_args(_normalize_global_args(argv))

    base_project_root = Path(args.project_root).resolve()
    mode = Mode(apply=bool(args.apply))

    # Auto-select a tighter project root when requested.
    target_hint: Path = base_project_root
    if getattr(args, "file", None):
        target_hint = (base_project_root / args.file).resolve()
    elif getattr(args, "src", None):
        target_hint = (base_project_root / args.src).resolve()

    project_root = base_project_root
    if args.auto_project_root:
        project_root = _auto_project_root(target_hint, fallback_root=base_project_root)
        if project_root != base_project_root:
            print(f"Auto project root: {project_root}")

    scan_roots = args.scan_roots
    if args.auto_scan_roots and scan_roots is None:
        needles: list[str] = []

        # Symbol name is the best signal for symbol renames.
        if getattr(args, "symbol", None):
            needles.append(str(args.symbol))

        # For module moves/renames, the module basename is a weak signal.
        if getattr(args, "src", None):
            try:
                needles.append(Path(args.src).stem)
            except Exception:
                pass

        # Add a more specific signal: dotted module path for --file/--src.
        rel_for_mod = None
        if getattr(args, "file", None):
            try:
                abs_p = (base_project_root / args.file).resolve()
                rel_for_mod = str(abs_p.relative_to(project_root))
            except Exception:
                rel_for_mod = None
        elif getattr(args, "src", None):
            try:
                abs_p = (base_project_root / args.src).resolve()
                rel_for_mod = str(abs_p.relative_to(project_root))
            except Exception:
                rel_for_mod = None
        if rel_for_mod:
            mod = _file_to_dotted_module_name(project_root, rel_for_mod)
            if mod:
                needles.append(mod)
                if getattr(args, "symbol", None):
                    needles.append(f"{mod}:{args.symbol}")
                    needles.append(f"{mod}.{args.symbol}")

        if needles:
            scan_roots = _auto_scan_roots(project_root, needles=needles, include_paths=[target_hint])
            print("Auto scan roots: " + ", ".join(scan_roots))

    # If project_root was auto-adjusted, rewrite file/src args to be relative to it.
    normalized_file = getattr(args, "file", None)
    if normalized_file:
        try:
            abs_file = (base_project_root / normalized_file).resolve()
            normalized_file = str(abs_file.relative_to(project_root))
        except Exception:
            pass

    normalized_src = getattr(args, "src", None)
    if normalized_src:
        try:
            abs_src = (base_project_root / normalized_src).resolve()
            normalized_src = str(abs_src.relative_to(project_root))
        except Exception:
            pass

    if args.cmd == "move-module":
        _move_module(
            project_root=project_root,
            src=normalized_src or args.src,
            dest_package=args.dest_package,
            scan_roots=scan_roots,
            mode=mode,
        )
        return 0

    if args.cmd == "rename-module":
        _rename_module(
            project_root=project_root,
            src=normalized_src or args.src,
            new_name=args.new_name,
            scan_roots=scan_roots,
            mode=mode,
        )
        return 0

    if args.cmd == "rename-symbol":
        _rename_symbol(
            project_root=project_root,
            file=normalized_file or args.file,
            new_name=args.new_name,
            offset=args.offset,
            pattern=args.pattern,
            occurrence=args.occurrence,
            group=args.group,
            symbol=args.symbol,
            context=args.context,
            scan_roots=scan_roots,
            mode=mode,
        )
        return 0

    if args.cmd == "extract-function":
        _extract(
            project_root=project_root,
            file=normalized_file or args.file,
            new_name=args.new_name,
            start_line=args.start_line,
            end_line=args.end_line,
            kind="function",
            mode=mode,
        )
        return 0

    if args.cmd == "extract-method":
        _extract(
            project_root=project_root,
            file=normalized_file or args.file,
            new_name=args.new_name,
            start_line=args.start_line,
            end_line=args.end_line,
            kind="method",
            mode=mode,
        )
        return 0

    if args.cmd == "inline-variable":
        _inline(
            project_root=project_root,
            file=normalized_file or args.file,
            offset=args.offset,
            pattern=args.pattern,
            occurrence=args.occurrence,
            group=args.group,
            symbol=args.symbol,
            context=args.context,
            kind="variable",
            mode=mode,
        )
        return 0

    if args.cmd == "inline-method":
        _inline(
            project_root=project_root,
            file=normalized_file or args.file,
            offset=args.offset,
            pattern=args.pattern,
            occurrence=args.occurrence,
            group=args.group,
            symbol=args.symbol,
            context=args.context,
            kind="method",
            mode=mode,
        )
        return 0

    if args.cmd == "organize-imports":
        if args.files:
            files = [str(x) for x in args.files]
        elif args.roots:
            files = _collect_py_files(project_root, [str(x) for x in args.roots])
        else:
            _die("organize-imports requires --files or --roots")
        _organize_imports(project_root=project_root, files=files, mode=mode)
        return 0

    if args.cmd == "batch":
        _run_batch(project_root=project_root, map_file=Path(args.map).resolve(), mode=mode)
        return 0

    if args.cmd == "self-test":
        _self_test()
        return 0

    _die(f"Unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
