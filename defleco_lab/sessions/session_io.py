from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2

from defleco_lab.camera.frame_packet import FramePacket

FIELDS = (
    "frame_id",
    "host_timestamp_ns",
    "camera_timestamp",
    "block_id",
    "exposure_us",
    "gain",
    "width",
    "height",
    "pixel_format",
    "estimated_motion_px",
    "motion_quality",
    "cumulative_position_px",
    "source",
    "filename",
)


class SessionRecorder:
    def __init__(self, root: Path, notes: str = "") -> None:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        self.path = Path(root) / f"session_{stamp}.partial"
        self.frames = self.path / "frames"
        self.results = self.path / "results"
        self.frames.mkdir(parents=True)
        self.results.mkdir()
        self.notes = notes
        self.rows: list[dict[str, object]] = []
        self._write_manifest(False)

    def append(self, packet: FramePacket) -> None:
        filename = f"frame_{len(self.rows):06d}.png"
        target = self.frames / filename
        if not cv2.imwrite(str(target), packet.image):
            raise OSError(f"Could not write {target}")
        row = packet.public_metadata()
        row["filename"] = filename
        self.rows.append(row)
        self._write_metadata()

    def update_motion(
        self,
        frame_id: int,
        estimated_motion_px: float,
        motion_quality: float,
        cumulative_position_px: float,
    ) -> None:
        for row in reversed(self.rows):
            if int(row["frame_id"]) == int(frame_id):
                row["estimated_motion_px"] = estimated_motion_px
                row["motion_quality"] = motion_quality
                row["cumulative_position_px"] = cumulative_position_px
                self._write_metadata()
                return

    def _write_metadata(self) -> None:
        target = self.path / "metadata.csv"
        temporary = self.path / "metadata.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
            stream.flush()
        temporary.replace(target)

    def close(self) -> Path:
        self._write_manifest(True)
        final = self.path.with_name(self.path.name.removesuffix(".partial"))
        self.path.rename(final)
        self.path = final
        return final

    def _write_manifest(self, complete: bool) -> None:
        data = {
            "schema_version": 1,
            "app": "Defleco LAB",
            "complete": complete,
            "updated_utc": datetime.now(UTC).isoformat(),
            "notes": self.notes,
            "frame_count": len(self.rows),
            "source": "sanitized",
        }
        target = self.path / "session.json"
        temporary = self.path / "session.json.tmp"
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(target)


def load_session(path: Path) -> list[FramePacket]:
    path = Path(path)
    with (path / "metadata.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    packets: list[FramePacket] = []
    for row in rows:
        image = cv2.imread(str(path / "frames" / row["filename"]), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(row["filename"])
        packets.append(FramePacket.from_metadata(image, row, source="replay"))
    return packets
