import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_colmap_fused_ply(path):
    with Path(path).open("rb") as f:
        vertex_count = None
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("Unexpected EOF while reading PLY header")
            text = line.decode("ascii").strip()
            if text.startswith("element vertex"):
                vertex_count = int(text.split()[-1])
            if text == "end_header":
                break

        dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("nx", "<f4"),
                ("ny", "<f4"),
                ("nz", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )
        data = np.fromfile(f, dtype=dtype, count=vertex_count)

    points = np.stack([data["x"], data["y"], data["z"]], axis=1)
    colors = np.stack([data["red"], data["green"], data["blue"]], axis=1).astype(np.float32) / 255.0
    return points, colors


def main():
    ply_path = Path("data/colmap/dense/fused.ply")
    out_dir = Path("outputs/colmap")
    out_dir.mkdir(parents=True, exist_ok=True)

    points, colors = read_colmap_fused_ply(ply_path)
    step = max(1, len(points) // 45000)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        points[::step, 0],
        points[::step, 1],
        points[::step, 2],
        c=colors[::step],
        s=0.35,
        linewidths=0,
    )
    ax.set_axis_off()
    ax.view_init(elev=12, azim=-72)
    ax.set_box_aspect(
        (
            np.ptp(points[:, 0]) + 1e-6,
            np.ptp(points[:, 1]) + 1e-6,
            np.ptp(points[:, 2]) + 1e-6,
        )
    )
    plt.tight_layout(pad=0)
    preview_path = out_dir / "dense_fused_preview.png"
    fig.savefig(preview_path, dpi=220)
    plt.close(fig)

    summary = {
        "dense_fused_vertices": int(len(points)),
        "fused_ply": ply_path.as_posix(),
        "preview": preview_path.as_posix(),
        "bounds_min": points.min(axis=0).tolist(),
        "bounds_max": points.max(axis=0).tolist(),
    }
    summary_path = out_dir / "colmap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
