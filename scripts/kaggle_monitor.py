#!/usr/bin/env python3
"""Resilient monitor for Kaggle kernel execution and automated output downloader."""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

KERNEL_SLUG = "saurabmmshra/nl2ms-phase-transitions"
OUTPUT_DIR = Path("/home/saurab/Documents/NL2Ms/NL2Ms/kaggle_outputs")
POLL_INTERVAL_SECONDS = 60
MAX_MONITOR_HOURS = 12.5


def log(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def get_kernel_status() -> Optional[str]:
    for _ in range(3):
        try:
            res = subprocess.run(
                ["kaggle", "kernels", "status", KERNEL_SLUG],
                capture_output=True,
                text=True,
                timeout=60,
            )
            output = res.stdout.strip()
            if 'status "' in output:
                return output.split('status "')[1].rstrip('"')
            if output.startswith("saurabmmshra/"):
                return output
        except Exception:
            time.sleep(3)
    return None


def get_latest_logs(last_line_count: int = 15) -> str:
    try:
        res = subprocess.run(
            ["kaggle", "kernels", "logs", KERNEL_SLUG],
            capture_output=True,
            text=True,
            timeout=45,
        )
        lines = res.stdout.strip().splitlines()
        if not lines:
            return ""
        return "\n".join(lines[-last_line_count:])
    except Exception as exc:
        return f"(Log fetch delayed: {exc})"


def download_outputs() -> bool:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Downloading outputs from Kaggle to {OUTPUT_DIR}...")
    try:
        res = subprocess.run(
            ["kaggle", "kernels", "output", KERNEL_SLUG, "-p", str(OUTPUT_DIR), "--force"],
            capture_output=True,
            text=True,
            timeout=900,
        )
        log(f"Download output:\n{res.stdout}\n{res.stderr}")
        log("Downloaded files summary:")
        total_files = 0
        for p in OUTPUT_DIR.rglob("*"):
            if p.is_file():
                total_files += 1
                log(f"  - {p.relative_to(OUTPUT_DIR)} ({p.stat().st_size / (1024 * 1024):.2f} MB)")
        log(f"Total downloaded files: {total_files}")
        return res.returncode == 0
    except Exception as exc:
        log(f"Failed to download outputs: {exc}")
        return False


def main() -> int:
    log(f"Starting resilient monitor for Kaggle kernel: {KERNEL_SLUG}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    last_status = None
    prev_log_snippet = ""

    while True:
        elapsed_sec = time.time() - start_time
        elapsed_hr = elapsed_sec / 3600.0

        if elapsed_hr >= MAX_MONITOR_HOURS:
            log(f"Reached maximum monitoring time ({MAX_MONITOR_HOURS}h). Proceeding to output download.")
            break

        status = get_kernel_status()
        if status is None:
            log("Note: Kaggle API status request timed out, will retry next poll interval.")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        if status != last_status:
            log(f"Status changed -> {status} (Elapsed: {elapsed_hr:.2f}h)")
            last_status = status

        logs = get_latest_logs(10)
        if logs and logs != prev_log_snippet:
            log(f"--- Latest Logs snippet ---\n{logs}\n---------------------------")
            prev_log_snippet = logs

        # Check genuine Kaggle terminal statuses
        if "KernelWorkerStatus.COMPLETE" in status or status == "COMPLETE":
            log("Kernel run completed successfully on Kaggle!")
            break
        elif any(term in status for term in [
            "KernelWorkerStatus.ERROR",
            "KernelWorkerStatus.CANCEL",
            "KernelWorkerStatus.FAILED",
            "KernelWorkerStatus.KILLED",
        ]):
            log(f"Kernel reached terminal status: {status}")
            break

        time.sleep(POLL_INTERVAL_SECONDS)

    log("Kernel run finished. Downloading outputs now...")
    download_outputs()
    log("Monitor task finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
