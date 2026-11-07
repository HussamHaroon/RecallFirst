"""AST-guided chunking of repositories into code chunks (Research.md 5.A).

Uses tree-sitter (via tree-sitter-language-pack) to split source files into
logical units: top-level functions, classes, and methods. Files that fail to
parse (or have no functions) become whole-file chunks, capped in size.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from tree_sitter_language_pack import get_parser

# node types that become chunks, per language (python shown; extend as needed)
CHUNK_NODE_TYPES = {
    "python": {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "class_declaration", "method_definition"},
    "typescript": {"function_declaration", "class_declaration", "method_definition"},
}

# Fallback: max chars per whole-file chunk
MAX_FILE_CHARS = 4000


@dataclass
class CodeChunk:
    repo_path: str   # path relative to repo root
    name: str        # qualified name, e.g. "ClassName.method"
    text: str
    line_start: int  # 1-based


def iter_source_files(repo_dir: str, exts=(".py",)) -> list[str]:
    out = []
    skip = {".git", "node_modules", "__pycache__", ".venv", "venv", "docs", "tests"}
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in skip]
        for f in files:
            if f.endswith(exts):
                out.append(os.path.join(root, f))
    return sorted(out)


def chunk_file(path: str, repo_dir: str, language: str = "python") -> list[CodeChunk]:
    rel = os.path.relpath(path, repo_dir).replace("\\", "/")
    src = open(path, encoding="utf-8", errors="replace").read()
    if not src.strip():
        return []
    lang_types = CHUNK_NODE_TYPES.get(language)
    if lang_types is None:
        return _file_chunk(rel, src)
    try:
        parser = get_parser(language)
        tree = parser.parse(src.encode("utf-8"))
    except Exception:
        return _file_chunk(rel, src)

    chunks: list[CodeChunk] = []

    def walk(node, scope: str):
        for child in node.children:
            if child.type in lang_types:
                # only top-level defs; methods handled when we recurse into class bodies
                is_method = bool(scope) and child.type == "function_definition"
                name = _decl_name(child, src) or "<anon>"
                qual = f"{scope}.{name}" if scope else name
                text = src[child.start_byte : child.end_byte]
                if len(text.strip()) > 0:
                    chunks.append(CodeChunk(rel, qual, text, child.start_point[0] + 1))
                if child.type == "class_definition":
                    walk(child, qual)
            else:
                walk(child, scope)

    walk(tree.root_node, "")
    if not chunks:
        return _file_chunk(rel, src)
    return chunks


def _decl_name(node, src: str) -> str | None:
    for child in node.children:
        if child.type == "identifier":
            return src[child.start_byte : child.end_byte]
    return None


def _file_chunk(rel: str, src: str) -> list[CodeChunk]:
    if len(src) > MAX_FILE_CHARS:
        src = src[:MAX_FILE_CHARS]
    return [CodeChunk(rel, "<file>", src, 1)]


def chunk_repo(repo_dir: str, language: str = "python") -> list[CodeChunk]:
    exts = {".py": (".py",), "javascript": (".js", ".ts", ".jsx", ".tsx")}
    files = iter_source_files(repo_dir, exts.get(language, (".py",)))
    out: list[CodeChunk] = []
    for f in files:
        out.extend(chunk_file(f, repo_dir, language))
    return out
