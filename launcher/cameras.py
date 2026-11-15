"""Camera list (for the dropdown) and a one-frame snapshot (for the preview), with rt's own code.

* Windows: rt/camsource.list_dshow_video_devices, ffmpeg's DirectShow list, in the same order as OpenCV's indices.
  Each camera has a device path; that is what goes into the config (identical webcams share a name).
* Linux (repo mode): /dev/v4l/by-id links, else /dev/video*.

A camera can be opened by one program at a time: while a role runs, its camera cannot be previewed.
"""
import subprocess
import sys
from pathlib import Path

from launcher.procs import windows_flags


def quiet_run(cmd, **kwargs):
    """subprocess.run without a console window flashing up: the tray app runs under pythonw.exe (no console), so a
    console program it starts (ffmpeg) would otherwise get a window of its own for a moment."""
    return subprocess.run(cmd, creationflags=windows_flags(visible=False), **kwargs)


def quiet_lister(ffmpeg):
    from rt.camsource import list_dshow_video_devices
    return list_dshow_video_devices(ffmpeg, runner=quiet_run)


def list_cameras(ffmpeg, lister=None, platform=None):
    """-> (cameras, error). cameras: [dict(index, name, device)] where device is what the config should hold."""
    platform = platform or sys.platform
    if platform == "win32":
        from rt.camsource import CameraError
        try:
            devices = (lister or quiet_lister)(ffmpeg)
        except CameraError as e:
            return [], str(e)
        return [dict(index=i, name=name, device=path or str(i)) for i, (name, path) in enumerate(devices)], None
    by_id = sorted(Path("/dev/v4l/by-id").glob("*")) if Path("/dev/v4l/by-id").exists() else []
    if by_id:
        return [dict(index=i, name=link.name, device=str(link)) for i, link in enumerate(by_id)], None
    nodes = sorted(Path("/dev").glob("video*"))
    return [dict(index=i, name=node.name, device=str(node)) for i, node in enumerate(nodes)], None


def snapshot_jpeg(device, ffmpeg, width=1920, height=1080, cv2_module=None, frames_to_skip=5):
    """Open the camera like rt does (rt/camsource.resolve_device: index or device path), grab one frame and return
    it as JPEG bytes (downscaled to 960 px wide). Raises RuntimeError with a pt-BR message on failure."""
    import cv2 as real_cv2
    cv2 = cv2_module or real_cv2
    from rt.camsource import CameraError, default_backend, resolve_device
    backend = default_backend()
    try:
        target = resolve_device(str(device), backend, ffmpeg, lister=quiet_lister)
    except CameraError as e:
        raise RuntimeError("Câmera não encontrada: %s" % e) from e
    api = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "v4l2": cv2.CAP_V4L2}.get(backend, cv2.CAP_ANY)
    cap = cv2.VideoCapture(target, api)
    try:
        if not cap.isOpened():
            raise RuntimeError("Não foi possível abrir a câmera (ela está em uso por um papel em execução?)")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        frame = None
        for _ in range(frames_to_skip + 1):           # the first frames are often dark (auto exposure settling)
            ok, image = cap.read()
            if ok and image is not None:
                frame = image
        if frame is None:
            raise RuntimeError("A câmera abriu mas não entregou imagem.")
    finally:
        cap.release()
    h, w = frame.shape[:2]
    if w > 960:
        frame = cv2.resize(frame, (960, int(round(h * 960 / w))))
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("Falha ao codificar a imagem.")
    return jpeg.tobytes(), (w, h)
