from __future__ import annotations

"""Isolated PolyFit engine process.

The GUI supervises this script so native solver calls cannot monopolize the GUI's
Python interpreter/GIL. Events are newline-delimited JSON on stdout.
"""

import json
import sys
import traceback
from pathlib import Path

import numpy as np

from pipeline import run_reconstruction
from workspace import mesh_work_dir

def emit(event: dict) -> None:

    print(json.dumps(event, ensure_ascii=True, default=_json_default), flush=True)

def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")

def main() -> int:
    if len(sys.argv) != 2:
        emit({"type": "error", "message": "worker.py requires a job configuration JSON path."})
        return 2

    try:
        config_path = Path(sys.argv[1]).resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        work_dir = Path(config.get("work_dir") or mesh_work_dir(config["input_path"]))
        work_dir.mkdir(parents=True, exist_ok=True)
        config["work_dir"] = str(work_dir)
        mesh_preview_path = work_dir / "after_preview.npz"
        mesh_preview_path.unlink(missing_ok=True)
        live_mesh_preview_written = False
        stage_preview_sequence = 0

        def preview_points(points, colors):
            preview_path = work_dir / "before_preview.npz"
            np.savez(
                preview_path,
                points=np.asarray(points, dtype=np.float32),
                colors=np.asarray(colors, dtype=np.float32),
            )
            emit({"type": "preview_points", "path": str(preview_path)})

        def preview_stage_mesh(vertices, faces, label, highlight_segments=None):
            """Write compact edge-only geometry for algorithm drawing."""
            nonlocal stage_preview_sequence
            try:
                stage_preview_sequence += 1
                preview_path = work_dir / f"processing_preview_{stage_preview_sequence:04d}.npz"
                stage_vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
                if isinstance(faces, dict):
                    raw_triangles = np.asarray(faces.get("triangles", []), dtype=np.int64).reshape(-1, 3)
                    raw_quads = np.asarray(faces.get("quads", []), dtype=np.int64).reshape(-1, 4)
                else:
                    raw_triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
                    raw_quads = np.empty((0, 4), dtype=np.int64)

                triangle_valid = (raw_triangles >= 0).all(axis=1) & (raw_triangles < len(stage_vertices)).all(axis=1)
                quad_valid = (raw_quads >= 0).all(axis=1) & (raw_quads < len(stage_vertices)).all(axis=1)
                triangle_indices = np.flatnonzero(triangle_valid)
                quad_indices = np.flatnonzero(quad_valid)
                stage_faces = raw_triangles[triangle_indices]
                stage_quads = raw_quads[quad_indices]

                total_polygons = len(stage_faces) + len(stage_quads)
                if total_polygons > 180_000:
                    tri_budget = int(round(180_000 * len(stage_faces) / total_polygons))
                    quad_budget = 180_000 - tri_budget
                    tri_take = np.linspace(0, len(stage_faces) - 1, tri_budget, dtype=np.int64) if tri_budget and len(stage_faces) > tri_budget else np.arange(len(stage_faces), dtype=np.int64)
                    quad_take = np.linspace(0, len(stage_quads) - 1, quad_budget, dtype=np.int64) if quad_budget and len(stage_quads) > quad_budget else np.arange(len(stage_quads), dtype=np.int64)
                    stage_faces = stage_faces[tri_take]
                    stage_quads = stage_quads[quad_take]

                if len(stage_faces) + len(stage_quads) == 0:
                    return
                referenced = np.concatenate((stage_faces.reshape(-1), stage_quads.reshape(-1)))
                used, inverse = np.unique(referenced, return_inverse=True)
                stage_vertices = stage_vertices[used]
                tri_count = stage_faces.size
                stage_faces = inverse[:tri_count].reshape(-1, 3).astype(np.uint32, copy=False)
                stage_quads = inverse[tri_count:].reshape(-1, 4).astype(np.uint32, copy=False)

                payload = {
                    "vertices": stage_vertices,
                    "faces": stage_faces,
                    "quads": stage_quads,
                }
                if highlight_segments is not None:
                    segments = np.asarray(highlight_segments, dtype=np.float32).reshape(-1, 2, 3)
                    if len(segments) > 12000:
                        take = np.linspace(0, len(segments) - 1, 12000, dtype=np.int64)
                        segments = segments[take]
                    payload["highlight_segments"] = segments
                np.savez(preview_path, **payload)
                emit({"type": "stage_mesh", "path": str(preview_path), "label": str(label)})
            except Exception as exc:
                emit({"type": "log", "message": f"Live processing-mesh preview could not be saved; reconstruction will continue: {exc}"})

        def preview_mesh(vertices, faces, colors=None):
            nonlocal live_mesh_preview_written
            try:
                payload = {
                    "vertices": np.asarray(vertices, dtype=np.float32),
                    "faces": np.asarray(faces, dtype=np.uint32),
                }
                if colors is not None:
                    payload["colors"] = np.asarray(colors, dtype=np.float32)
                np.savez(mesh_preview_path, **payload)
                live_mesh_preview_written = True
                emit({"type": "preview_mesh", "path": str(mesh_preview_path)})
            except Exception as exc:

                emit({"type": "log", "message": f"Live mesh preview could not be saved; reconstruction will continue: {exc}"})

        result = run_reconstruction(
            config,
            progress_cb=lambda value, message: emit({"type": "progress", "value": int(value), "message": str(message)}),
            log_cb=lambda message: emit({"type": "log", "message": str(message)}),
            preview_cb=preview_points,
            mesh_preview_cb=preview_mesh,
            stage_preview_cb=preview_stage_mesh,
        )

        preview = result.pop("mesh_preview")

        if not live_mesh_preview_written:
            np.savez(
                mesh_preview_path,
                vertices=np.asarray(preview["vertices"], dtype=np.float32),
                faces=np.asarray(preview["faces"], dtype=np.uint32),
            )
        result["mesh_preview_path"] = str(mesh_preview_path)
        emit({"type": "completed", "result": result})
        return 0
    except Exception as exc:
        emit({"type": "error", "message": f"{exc}\n\n--- Full traceback ---\n{traceback.format_exc()}"})
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
