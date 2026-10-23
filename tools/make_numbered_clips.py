#!/usr/bin/env python3
"""Synthetic high-frame-rate test clips whose every frame carries its own number.

Each 1920x1080 frame n shows:
- a black band across the top 120 rows with 16 cells of 120 px; cell b (from
  the left) holds a white 100x100 square when bit b of n is set. The plugins'
  temporary logs sample the cell centres (x = 120*b + 60, y = 60 from the top)
  to read back which source frame the host actually delivered;
- n in large digits in the centre and "<rate> fps  frame n" below it;
- an 8 px white vertical bar at x = 8*n mod 1920, so blending is visible as a
  smeared bar.
Encoded as H.264 High, full-range 4:2:0 BT.709, like the recordings.
"""
import argparse
from pathlib import Path
import shutil
import subprocess

FONT = "C\\:/Windows/Fonts/arialbd.ttf"


def build_filter(rate):
    parts = ["drawbox=x=0:y=0:w=1920:h=120:color=black:t=fill"]
    for bit in range(16):
        parts.append(f"drawbox=x={120 * bit + 10}:y=10:w=100:h=100:color=white:t=fill"
                     f":enable='mod(floor(n/{2 ** bit})\\,2)'")
    parts.append(f"drawtext=fontfile='{FONT}':text='%{{frame_num}}':fontsize=360:fontcolor=white"
                 ":x=(w-tw)/2:y=(h-th)/2")
    parts.append(f"drawtext=fontfile='{FONT}':text='{rate} fps  frame %{{frame_num}}':fontsize=64"
                 ":fontcolor=white:x=(w-tw)/2:y=h-160")
    return ",".join(parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rates", default="240,600")
    parser.add_argument("--seconds", type=int, default=6)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    args = parser.parse_args(argv)
    for rate in (int(r) for r in args.rates.split(",")):
        frames = rate * args.seconds
        output = args.output_dir / f"numbered-{rate}fps-{args.seconds}s-1080p.mp4"
        graph = (f"color=c=0x303030:s=1920x1080:r={rate}:d={args.seconds}[bg];"
                 f"color=c=white:s=8x960:r={rate}:d={args.seconds}[bar];"
                 f"[bg][bar]overlay=x='mod(n*8\\,1920)':y=120:eval=frame,{build_filter(rate)}[v]")
        subprocess.run([args.ffmpeg, "-v", "error", "-y", "-filter_complex", graph, "-map", "[v]",
                        "-frames:v", str(frames), "-c:v", "libx264", "-preset", "medium", "-crf", "12",
                        "-profile:v", "high", "-g", str(2 * rate), "-pix_fmt", "yuvj420p",
                        "-color_range", "pc", "-colorspace", "bt709", "-color_primaries", "bt709",
                        "-color_trc", "bt709", "-r", str(rate), "-movflags", "+faststart", str(output)],
                       check=True)
        print(f"wrote {output} ({frames} frames at {rate} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
