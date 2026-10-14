#!/usr/bin/env python3
"""Monitor Kaggle kernel execution and automatically download outputs upon completion."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

KERNEL_SLUG = "saurabmmshra/nl2ms-phase-transitions"
OUTPUT_DIR = Path("/home/saurab/Documents/NL2Ms/NL2Ms/kaggle_outputs")
POLL_INTERVAL_SECONDS = 45
MAX_MONITOR_HOURS = 12.5


def log(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def get_kernel_status() -> str:
    try:
        res = subprocess.run(
            ["kaggle", "kernels", "status", KERNEL_SLUG],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = res.stdout.strip()
        # Parse status: e.g. 'saurabmmshra/nl2ms-phase-transitions has status "KernelWorkerStatus.RUNNING"'
        if 'status "' in output:
            status = output.split('status "')[1].rstrip('"')
            return status
        return output
    except Exception as exc:
        return f"ERROR_FETCHING_STATUS: {exc}"


def get_latest_logs(last_line_count: int = 15) -> str:
    try:
        res = subprocess.run(
            ["kaggle", "kernels", "logs", KERNEL_SLUG],
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines = res.stdout.strip().splitlines()
        if not lines:
            return ""
        return "\n".join(lines[-last_line_count:])
    except Exception as exc:
        return f"(Could not fetch logs: {exc})"


def download_outputs() -> bool:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Downloading outputs to {OUTPUT_DIR}...")
    try:
        res = subprocess.run(
            ["kaggle", "kernels", "output", KERNEL_SLUG, "-p", str(OUTPUT_DIR), "--force"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        log(f"Download output:\n{res.stdout}\n{res.stderr}")
        log("Downloaded files:")
        for p in OUTPUT_DIR.rglob("*"):
            if p.is_file():
                log(f"  - {p.relative_to(OUTPUT_DIR)} ({p.stat().st_size / (1024 * 1024):.2f} MB)")
        return res.returncode == 0
    except Exception as exc:
        log(f"Failed to download outputs: {exc}")
        return False


def main() -> int:
    log(f"Starting monitor for Kaggle kernel: {KERNEL_SLUG}")
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
        if status != last_status:
            log(f"Status changed -> {status} (Elapsed: {elapsed_hr:.2f}h)")
            last_status = status

        logs = get_latest_logs(10)
        if logs and logs != prev_log_snippet:
            log(f"--- Latest Logs snippet ---\n{logs}\n---------------------------")
            prev_log_snippet = logs

        # Check terminal status
        if "COMPLETE" in status.upper():
            log("Kernel run completed successfully!")
            break
        elif any(term in status.upper() for term in ["ERROR", "CANCEL", "FAILED", "KILLED"]):
            log(f"Kernel terminated with status: {status}")
            break

        time.sleep(POLL_INTERVAL_SECONDS)

    log("Run terminated or completed. Initiating final output fetch...")
    download_outputs()
    log("Monitor task finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
