"""Decode throughput of a stream: CPU vs -hwaccel cuda, full rate vs an fps filter.

usage: python tools/bench/decode.py <video> [start_s=600] [dur_s=300] [threads=0]
Two sinks per variant:
  null : ffmpeg -f null (pure decode + filter cost)
  pipe : rawvideo bgr24 piped into Python and wrapped as numpy frames (what a detector loop actually pays)
Prints source-seconds decoded per wall second (x real time) and output frames per second.
"""
import os
import shutil
import subprocess
import sys
import time

import numpy as np

WINGET = r"C:\Users\baja\AppData\Local\Microsoft\WinGet\Links"
if not shutil.which("ffmpeg") and os.path.isdir(WINGET):
    os.environ["PATH"] = WINGET + os.pathsep + os.environ["PATH"]


def run(video, start, dur, hw, fps, sink, threads, w=1280, h=720):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if hw:
        cmd += ["-hwaccel", "cuda"]
    if threads:
        cmd += ["-threads", str(threads)]
    cmd += ["-ss", str(start), "-t", str(dur), "-i", video, "-an"]
    if fps:
        cmd += ["-vf", "fps=%g" % fps]
    if sink == "null":
        cmd += ["-f", "null", "-"]
        t = time.time()
        subprocess.run(cmd, check=True)
        dt = time.time() - t
        n = dur * (fps or 30)
    else:
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        fs = w * h * 3
        t = time.time()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=fs * 4)
        n = 0
        while True:
            b = p.stdout.read(fs)
            if len(b) < fs:
                break
            np.frombuffer(b, np.uint8).reshape(h, w, 3)
            n += 1
        p.wait()
        dt = time.time() - t
    return dt, n


def main():
    video = sys.argv[1]
    start = float(sys.argv[2]) if len(sys.argv) > 2 else 600
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 300
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    print("video %s  segment %.0f+%.0f s  threads=%s" % (os.path.basename(video), start, dur, threads or "auto"))
    print("%-5s %-6s %-5s %8s %10s %10s" % ("dec", "fps", "sink", "wall_s", "x_realtime", "out_fps"))
    for sink in ["null", "pipe"]:
        for fps in [None, 4, 2]:
            for hw in [False, True]:
                dt, n = run(video, start, dur, hw, fps, sink, threads)
                print("%-5s %-6s %-5s %8.1f %10.1f %10.1f" % ("cuda" if hw else "cpu", fps or "full", sink, dt,
                                                           dur / dt, n / dt), flush=True)


if __name__ == "__main__":
    main()
