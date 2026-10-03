"""Serve the pinned native Shinka interface on loopback for selected archives."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
import re
import sqlite3
from urllib.parse import parse_qs, unquote, urlparse

from shinka.webui import visualization

ROOT = Path(__file__).resolve().parents[1]
STATIC = {"/", "/index.html", "/viz_tree.html", "/compare.html", "/favicon.png", "/sakana.jpg"}
API = {"/get_programs", "/get_programs_summary", "/get_program_count", "/get_program_details",
       "/get_meta_files", "/get_meta_content", "/download_meta_pdf", "/get_plots",
       "/get_system_prompts", "/get_database_stats"}


class ReadOnlySQLite:
    """Native quick-stat helpers also need read-only handles, including WAL reads."""

    def __getattr__(self, name):
        return getattr(sqlite3, name)

    def connect(self, database, *args, **kwargs):
        kwargs["uri"] = True
        connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", *args, **kwargs)
        connection.execute("PRAGMA query_only = ON")
        return connection


class LocalArchiveHandler(visualization.DatabaseRequestHandler):
    """Retain native views and read-only database queries within selected archives."""

    def __init__(self, *args, search_root, archives, **kwargs):
        self.archives = tuple(archives)
        super().__init__(*args, search_root=str(search_root),
                         directory=str(Path(visualization.__file__).parent), **kwargs)

    def allowed(self, path):
        resolved = Path(path).resolve()
        return any(resolved.is_relative_to(archive) for archive in self.archives)

    def do_GET(self):
        host = self.headers.get("Host", "").split(":", 1)[0]
        if host not in {"127.0.0.1", "localhost"}:
            self.send_error(403, "Use the loopback address")
            return
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path in API:
            databases = query.get("db_path", [])
            if (len(databases) != 1 or Path(databases[0]).is_absolute()
                    or not self.allowed(Path(self.search_root) / databases[0])
                    or Path(databases[0]).suffix not in {".sqlite", ".db"}):
                self.send_error(403, "Database is outside the selected archives")
                return
            for name in ("generation", "processed_count"):
                if name in query and (len(query[name]) != 1 or not re.fullmatch(r"[0-9]+", query[name][0])):
                    self.send_error(400, "Expected a nonnegative generation index")
                    return
        elif parsed.path.startswith("/plot_file/"):
            relative = Path(unquote(parsed.path[len("/plot_file/"):]))
            if (relative.is_absolute() or not self.allowed(Path(self.search_root) / relative)
                    or relative.suffix.lower() not in {".png", ".gif", ".jpg", ".jpeg"}):
                self.send_error(403, "Plot is outside the selected archives")
                return
        elif parsed.path not in STATIC | {"/list_databases"}:
            self.send_error(404, "Not found")
            return
        super().do_GET()

    def do_HEAD(self):
        self.send_error(405, "Use GET")

    def handle_list_databases(self):
        files = []
        for archive in self.archives:
            if not archive.is_dir():
                continue
            for database in sorted(archive.rglob("*")):
                if (database.suffix not in {".db", ".sqlite"} or not database.is_file()
                        or database.name in {"prompts.db", "prompts.sqlite"} or not self.allowed(database)):
                    continue
                relative = database.relative_to(self.search_root).as_posix()
                files.append({"path": relative, "actual_path": relative,
                              "name": database.parent.relative_to(self.search_root).as_posix(),
                              "task": archive.parent.name})
        self.send_json_response(files)

    def _read_failure_json(self, failure_json_path):
        if failure_json_path and not self.allowed(Path(self.search_root) / failure_json_path):
            return None
        return super()._read_failure_json(failure_json_path)

    def _resolve_failed_node_code_path(self, details, failure_payload):
        path = super()._resolve_failed_node_code_path(details, failure_payload)
        return path if path is None or self.allowed(path) else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--archive", type=Path, action="append", help="Selected result directory; repeatable")
    args = parser.parse_args(argv)
    results = (ROOT / "results").resolve()
    archives = [path.resolve() for path in args.archive] if args.archive else [
        results / "adaptive-shinka-20261003/shinka", results / "search-20261002-r1/shinka"]
    if not 1024 <= args.port <= 65535 or any(not path.is_relative_to(results) for path in archives):
        parser.error("Use an unprivileged port and archives inside this repository's results directory")
    handler = partial(LocalArchiveHandler, search_root=results, archives=archives)
    # Replace this module reference only; never alter the installed package or
    # sqlite3 itself. Native ProgramDatabase already uses mode=ro for UI reads.
    visualization.sqlite3 = ReadOnlySQLite()
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Native Shinka UI: http://127.0.0.1:{args.port}/", flush=True)
        for archive in archives:
            print(f"Selected archive: {archive.relative_to(results)}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
