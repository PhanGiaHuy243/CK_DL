"""Watch an existing pipeline and display one Windows notification when it finishes."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def now():
    return datetime.now(timezone.utc).isoformat()


def read_status(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def make_notification(state, directory):
    if state["status"] == "completed":
        target = Path(state.get("comparison", directory / "comparison/comparison.md")).resolve()
        if not target.is_relative_to(directory):
            raise ValueError("Report path is outside the pipeline directory")
        if not target.is_file():
            raise FileNotFoundError(f"Pipeline reported completion but report is missing: {target}")
        return {"title": "CHENAB — Đã hoàn tất", "outcome": "completed", "target": str(target),
                "message": "Đã hoàn tất Baseline, Augmented, đánh giá và so sánh.\n\n"
                           f"Báo cáo: {target}\n\nNhấn OK để mở báo cáo."}
    stage = str(state.get("stage", "unknown"))
    target = directory / "pipeline_status.json"
    return {"title": "CHENAB — Pipeline đã dừng", "outcome": state["status"], "target": str(target),
            "message": f"Pipeline chưa hoàn tất. Bước hiện tại: {stage}.\n\n"
                       f"Chi tiết: {str(state.get('error') or state.get('stop_reason') or state['status'])[:1500]}\n\n"
                       f"Log và trạng thái: {directory}\n\nNhấn OK để xem trạng thái."}


def show_notification(payload):
    import winsound
    try:
        winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
    except RuntimeError:
        pass  # A muted/unavailable sound device must not prevent the visual notification.
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    message_box = user32.MessageBoxW
    message_box.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
    message_box.restype = ctypes.c_int
    # Information, OK/Cancel, foreground, topmost. The dialog remains until acknowledged.
    response = message_box(None, payload["message"], payload["title"], 0x40 | 0x1 | 0x10000 | 0x40000)
    if response == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    if response == 1:
        target = Path(payload["target"])
        try:
            os.startfile(str(target))
        except OSError:
            os.startfile(str(target.parent))
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-file", default="outputs/matched/pipeline_status.json")
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--check-only", action="store_true", help="Check attachment without showing any notification")
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("Desktop notifications for this script require Windows")
    if args.interval < 1:
        raise ValueError("Polling interval must be at least one second")
    status_file = Path(args.status_file).resolve()
    directory = status_file.parent
    initial = read_status(status_file)
    controller_pid = int(initial["pid"])
    try:
        controller_created = psutil.Process(controller_pid).create_time()
    except psutil.NoSuchProcess:
        controller_created = None
    if args.check_only:
        print(json.dumps({"pipeline_status": initial["status"], "controller_pid": controller_pid,
                          "controller_exists": controller_created is not None,
                          "windows_notification_api": bool(ctypes.windll.user32.MessageBoxW)}, indent=2))
        return
    lock_path = directory / "notification_watcher.lock"
    lock = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(lock, str(os.getpid()).encode("ascii"))
    os.close(lock)
    watcher_path = directory / "notification_status.json"
    watcher = {"status": "watching", "pid": os.getpid(), "started_utc": now(),
               "pipeline_started_utc": initial["started_utc"], "controller_pid": controller_pid,
               "poll_interval_seconds": args.interval, "channel": "Windows popup and sound",
               "status_file": str(status_file)}

    def save():
        tmp = watcher_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(watcher, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(watcher_path)

    try:
        save()
        missing = 0
        while True:
            try:
                state = read_status(status_file)
            except (OSError, json.JSONDecodeError):
                time.sleep(args.interval)
                continue
            watcher["last_checked_utc"] = now()
            watcher["pipeline_status"] = state["status"]
            watcher["pipeline_stage"] = state.get("stage")
            if state.get("started_utc") != initial["started_utc"]:
                state = {**state, "status": "replaced", "error": "Lượt chạy đã được thay thế. Cần gắn lại bộ theo dõi thông báo."}
            if state["status"] in ("running", "paused"):
                try:
                    process = psutil.Process(controller_pid)
                    alive = process.is_running() and process.create_time() == controller_created
                except psutil.NoSuchProcess:
                    alive = False
                missing = 0 if alive else missing + 1
                if missing < 3:
                    save()
                    time.sleep(args.interval)
                    continue
                state = {**state, "status": "process_exited", "error": "Tiến trình điều phối đã dừng trước khi ghi kết quả hoàn tất."}
            payload = make_notification(state, directory)
            watcher.update(status="showing_notification", outcome=payload["outcome"], notified_utc=now())
            (directory / "notification.txt").write_text(payload["title"]+"\n\n"+payload["message"], encoding="utf-8")
            save()
            response = show_notification(payload)
            watcher.update(status="acknowledged", acknowledged_utc=now(), response=response)
            save()
            break
    except BaseException as exc:
        watcher.update(status="error", error=str(exc), finished_utc=now())
        save()
        raise
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
