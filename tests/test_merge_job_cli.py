"""merge_job_cli 的命令执行测试。"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "datasets/scripts"


def _load_cli():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            "merge_job_cli", SCRIPTS / "merge_job_cli.py"
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["merge_job_cli"] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(SCRIPTS))


def _write_registry(path: Path, *, map_rel: str | None = None) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "jobs": {
                    "pets": {
                        "prefix": "pets",
                        "candidates": ["cat", "dog"],
                        "src_rel": "downloads/raw",
                        "remap_out_rel": "downloads/remapped",
                        "map_rel": map_rel,
                        "split_ratios": [0.7, 0.2, 0.1],
                        "split_seed": 0,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_out(path: Path, *, sets: list[dict]) -> None:
    path.mkdir()
    (path / "data.yaml").write_text(
        yaml.safe_dump({"names": ["person", "cat", "dog"]}), encoding="utf-8"
    )
    (path / "merge_meta.json").write_text(
        json.dumps({"sets": sets}), encoding="utf-8"
    )


class TestMergeJobCli(unittest.TestCase):
    def test_list_reports_present_keep(self):
        cli = _load_cli()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            jobs = tmp_path / "jobs.yaml"
            out = tmp_path / "out"
            _write_registry(jobs)
            _write_out(out, sets=[{"prefix": "pets", "keep": ["cat", "dog"]}])
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                rc = cli.main(
                    [
                        "list",
                        "--jobs",
                        str(jobs),
                        "--data-root",
                        str(tmp_path / "data"),
                        "--out",
                        str(out),
                    ]
                )

        self.assertEqual(rc, 0)
        output = stdout.getvalue()
        self.assertIn("pets present keep=[cat, dog]", output)
        self.assertIn("candidates=[cat, dog]", output)

    def test_append_resolves_repo_relative_map_from_root(self):
        cli = _load_cli()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            jobs = tmp_path / "jobs.yaml"
            out = tmp_path / "out"
            _write_registry(jobs, map_rel="config/class_maps/example.yaml")
            _write_out(out, sets=[])

            with mock.patch.object(cli.subprocess, "run") as run:
                rc = cli.main(
                    [
                        "append",
                        "pets",
                        "--keep",
                        "cat",
                        "--jobs",
                        str(jobs),
                        "--data-root",
                        str(tmp_path / "data"),
                        "--out",
                        str(out),
                    ]
                )

        self.assertEqual(rc, 0)
        remap_command = run.call_args_list[0].args[0]
        map_index = remap_command.index("--map") + 1
        self.assertEqual(
            Path(remap_command[map_index]),
            ROOT / "config/class_maps/example.yaml",
        )

    def test_remove_reappend_failure_prints_recovery_command(self):
        cli = _load_cli()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            jobs = tmp_path / "jobs.yaml"
            out = tmp_path / "out"
            _write_registry(jobs)
            _write_out(out, sets=[{"prefix": "pets", "keep": ["cat", "dog"]}])
            error = subprocess.CalledProcessError(9, ["merge"])
            stderr = io.StringIO()

            with (
                mock.patch.object(
                    cli.subprocess, "run", side_effect=[mock.DEFAULT, error]
                ),
                contextlib.redirect_stderr(stderr),
            ):
                rc = cli.main(
                    [
                        "remove",
                        "pets",
                        "--drop",
                        "dog",
                        "--jobs",
                        str(jobs),
                        "--data-root",
                        str(tmp_path / "data"),
                        "--out",
                        str(out),
                    ]
                )

        self.assertNotEqual(rc, 0)
        self.assertIn("append pets --keep cat", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
