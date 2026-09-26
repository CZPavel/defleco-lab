from __future__ import annotations

from pathlib import Path
from time import perf_counter_ns

import cv2

from .frame_packet import FramePacket


def load_image_sequence(folder: Path) -> list[FramePacket]:
    paths = sorted(
        path
        for path in Path(folder).iterdir()
        if path.suffix.lower() in {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}
    )
    packets = []
    for index, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            packets.append(
                FramePacket(
                    image=image,
                    frame_id=index,
                    host_timestamp_ns=perf_counter_ns(),
                    pixel_format="loaded",
                    source="image-sequence",
                )
            )
    if not packets:
        raise ValueError("No readable images found in the selected folder")
    return packets
