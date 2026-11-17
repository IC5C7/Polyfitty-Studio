import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1] / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app import worker

class WorkerStagePreviewTests(unittest.TestCase):
    def test_worker_serializes_edge_geometry_without_face_colors(self):
        vertices = np.asarray((
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        ), dtype=np.float32)
        triangles = np.asarray(((0, 1, 2), (0, 2, 99)), dtype=np.int64)
        quads = np.asarray(((0, 1, 4, 3),), dtype=np.int64)
        highlights = np.asarray((((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),), dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "job.json"
            config_path.write_text(json.dumps({
                "input_path": str(root / "source.glb"),
                "work_dir": str(root / "work"),
            }), encoding="utf-8")

            def fake_run(config, progress_cb, log_cb, preview_cb, mesh_preview_cb, stage_preview_cb):
                stage_preview_cb(
                    vertices,
                    {"triangles": triangles, "quads": quads},
                    "TEST · LIVE DRAW",
                    highlight_segments=highlights,
                )
                return {"mesh_preview": {"vertices": vertices, "faces": np.asarray(((0, 1, 2),), dtype=np.uint32)}}

            stdout = io.StringIO()
            with patch.object(worker, "run_reconstruction", side_effect=fake_run), \
                 patch.object(sys, "argv", ["worker.py", str(config_path)]), \
                 contextlib.redirect_stdout(stdout):
                status = worker.main()

            self.assertEqual(status, 0)
            preview_files = sorted((root / "work").glob("processing_preview_*.npz"))
            self.assertEqual(len(preview_files), 1)
            with np.load(preview_files[0]) as preview:
                self.assertEqual(preview["faces"].shape, (1, 3))
                self.assertEqual(preview["quads"].shape, (1, 4))
                self.assertNotIn("face_colors", preview.files)
                np.testing.assert_array_equal(preview["highlight_segments"], highlights)
            self.assertIn('"type": "stage_mesh"', stdout.getvalue())

if __name__ == "__main__":
    unittest.main()
