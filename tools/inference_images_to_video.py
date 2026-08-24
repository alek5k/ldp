#!/usr/bin/env python3
"""Export image arrays in an inference rollout Zarr to per-episode MP4s.

The module intentionally keeps export and source-array deletion coupled: image
arrays are deleted only after every requested video has been written and
verified to be non-empty.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr


# CRF 35 is a deliberately compact HEVC profile for archival rollout review.
# Lower values preserve more detail but use more disk; 28 is a common medium
# quality starting point for HEVC.
HEVC_CRF = 35
HEVC_PRESET = "medium"


@dataclass(frozen=True)
class ImageVideoCompressionSummary:
    dataset_path: Path
    zarr_path: Path
    output_dir: Path
    image_keys: tuple[str, ...]
    episode_count: int
    source_images_deleted: bool


def disk_usage_bytes(path: str | Path) -> int:
    """Return filesystem space used by a file or directory, without links."""
    path = Path(path)
    try:
        stat = path.lstat()
    except FileNotFoundError:
        return 0
    if path.is_symlink():
        return 0
    if path.is_file():
        return int(getattr(stat, "st_blocks", 0) * 512 or stat.st_size)

    total = int(getattr(stat, "st_blocks", 0) * 512)
    for root, directories, files in os.walk(path, followlinks=False):
        directories[:] = [directory for directory in directories if not (Path(root) / directory).is_symlink()]
        for filename in files:
            file_path = Path(root) / filename
            try:
                file_stat = file_path.lstat()
            except FileNotFoundError:
                continue
            if not file_path.is_symlink():
                total += int(getattr(file_stat, "st_blocks", 0) * 512 or file_stat.st_size)
    return total


def _is_image_array(array) -> bool:
    if array.ndim != 4 or array.shape[-1] not in (1, 3, 4):
        return array.ndim == 4 and array.shape[1] in (1, 3, 4)
    return True


def image_keys_for_zarr(zarr_path: str | Path) -> tuple[str, ...]:
    root = zarr.open(str(zarr_path), mode="r")
    if "data" not in root:
        return ()
    return tuple(
        key for key in sorted(root["data"].keys())
        if "image" in key.lower() and _is_image_array(root["data"][key])
    )


def rollout_zarr_paths(dataset_path: str | Path) -> tuple[Path, ...]:
    """Find direct rollout Zarr stores with episode and image data."""
    dataset_path = Path(dataset_path)
    candidates = [dataset_path]
    candidates.extend(sorted(dataset_path.glob("*.zarr")))
    valid = []
    for candidate in candidates:
        if not candidate.is_dir() or not (candidate / "meta" / "episode_ends").exists():
            continue
        if image_keys_for_zarr(candidate):
            valid.append(candidate)
    return tuple(valid)


def _frame_to_bgr(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f"Expected image frame with 3 dimensions, got {frame.shape}.")
    if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.moveaxis(frame, 0, -1)
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    elif frame.shape[-1] == 4:
        frame = frame[..., :3]
    elif frame.shape[-1] != 3:
        raise ValueError(f"Expected 1, 3, or 4 image channels, got {frame.shape}.")
    if np.issubdtype(frame.dtype, np.floating) and frame.size:
        # Rollout images are normally normalised to [0, 1], but accept image
        # arrays already expressed in the usual [0, 255] range as well.
        if float(np.nanmax(frame)) <= 1.0:
            frame = frame * 255.0
    frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)
    return np.ascontiguousarray(frame[..., ::-1])


def _episode_bounds(episode_ends: np.ndarray, frame_count: int) -> tuple[tuple[int, int], ...]:
    episode_ends = np.asarray(episode_ends, dtype=np.int64).reshape(-1)
    if np.any(episode_ends <= 0) or np.any(np.diff(episode_ends) <= 0):
        raise ValueError("meta/episode_ends must be strictly increasing positive indices.")
    if len(episode_ends) and int(episode_ends[-1]) > frame_count:
        raise ValueError("Image array is shorter than meta/episode_ends.")
    starts = np.concatenate(([0], episode_ends[:-1]))
    return tuple((int(start), int(end)) for start, end in zip(starts, episode_ends))


class _HevcVideoWriter:
    """Stream BGR frames to FFmpeg's compact H.265 encoder."""

    def __init__(self, path: Path, frame: np.ndarray, fps: int):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to export compact H.265 videos.")
        height, width = frame.shape[:2]
        self.path = path
        self.process = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-loglevel", "error",
                "-f", "rawvideo",
                "-pixel_format", "bgr24",
                "-video_size", f"{width}x{height}",
                "-framerate", str(fps),
                "-i", "pipe:0",
                "-an",
                "-c:v", "libx265",
                "-preset", HEVC_PRESET,
                "-crf", str(HEVC_CRF),
                "-x265-params", "log-level=error",
                "-tag:v", "hvc1",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError(f"Video writer stdin was closed for {self.path}.")
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError as error:
            message = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
            raise RuntimeError(f"FFmpeg stopped while writing {self.path}: {message}") from error

    def release(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        returncode = self.process.wait()
        if returncode:
            message = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
            raise RuntimeError(f"FFmpeg failed to write {self.path}: {message}")


def _open_video_writer(path: Path, frame: np.ndarray, fps: int) -> _HevcVideoWriter:
    height, width = frame.shape[:2]
    if height % 2 or width % 2:
        raise ValueError(
            f"H.265 yuv420p export needs even image dimensions, got {width}x{height}."
        )
    return _HevcVideoWriter(path, frame, fps)


def compress_zarr_images_to_videos(
    dataset_path: str | Path,
    zarr_path: str | Path,
    *,
    fps: int = 10,
    delete_source_images: bool = False,
) -> ImageVideoCompressionSummary:
    """Export every episode and optionally remove verified source image arrays."""
    dataset_path = Path(dataset_path)
    zarr_path = Path(zarr_path)
    if fps < 1:
        raise ValueError("fps must be at least one.")
    image_keys = image_keys_for_zarr(zarr_path)
    if not image_keys:
        raise ValueError(f"No 4D image arrays were found in {zarr_path}.")

    root = zarr.open(str(zarr_path), mode="a" if delete_source_images else "r")
    if "meta" not in root or "episode_ends" not in root["meta"]:
        raise ValueError(f"{zarr_path} does not contain meta/episode_ends.")
    frame_count = min(root["data"][key].shape[0] for key in image_keys)
    bounds = _episode_bounds(root["meta"]["episode_ends"][:], frame_count)
    output_dir = dataset_path / "compressed_image_videos" / zarr_path.stem
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing video export: {output_dir}."
        )
    for key in image_keys:
        (output_dir / key).mkdir(parents=True, exist_ok=False)

    for episode_index, (start, end) in enumerate(bounds, start=1):
        writers: dict[str, _HevcVideoWriter] = {}
        try:
            for key in image_keys:
                first_frame = _frame_to_bgr(root["data"][key][start])
                video_path = output_dir / key / f"episode_{episode_index:04d}.mp4"
                writers[key] = _open_video_writer(video_path, first_frame, fps)
                writers[key].write(first_frame)
            for frame_index in range(start + 1, end):
                for key, writer in writers.items():
                    writer.write(_frame_to_bgr(root["data"][key][frame_index]))
        finally:
            for writer in writers.values():
                writer.release()
        if episode_index == 1 or episode_index % 10 == 0 or episode_index == len(bounds):
            print(f"Exported episode {episode_index}/{len(bounds)}")

    for key in image_keys:
        videos = sorted((output_dir / key).glob("*.mp4"))
        if len(videos) != len(bounds) or any(video.stat().st_size == 0 for video in videos):
            raise RuntimeError(f"Video verification failed for image key {key!r}.")

    if delete_source_images:
        for key in image_keys:
            del root["data"][key]
    return ImageVideoCompressionSummary(
        dataset_path=dataset_path,
        zarr_path=zarr_path,
        output_dir=output_dir,
        image_keys=image_keys,
        episode_count=len(bounds),
        source_images_deleted=delete_source_images,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Inference dataset directory.")
    parser.add_argument("--zarr", type=Path, help="Rollout Zarr to compress.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--delete-source-images", action="store_true")
    args = parser.parse_args()
    zarr_paths = rollout_zarr_paths(args.dataset)
    if not zarr_paths:
        raise SystemExit(f"No compatible rollout Zarr found under {args.dataset}.")
    if args.zarr is None:
        if len(zarr_paths) != 1:
            raise SystemExit("Pass --zarr when the dataset contains multiple rollout Zarr stores.")
        zarr_path = zarr_paths[0]
    else:
        zarr_path = args.zarr
    summary = compress_zarr_images_to_videos(
        args.dataset,
        zarr_path,
        fps=args.fps,
        delete_source_images=args.delete_source_images,
    )
    print(f"Videos written to {summary.output_dir}")
    if summary.source_images_deleted:
        print("Verified source image arrays were deleted.")


if __name__ == "__main__":
    main()
