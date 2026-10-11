#!/usr/bin/env python3
"""Independent thermal watchdog for the hand-tracking workload.

Runs as its own process (started by ~/handtrack/start.sh).  Once per second it
reads the hottest CPU/GPU thermal zone; if that stays >= WATCHDOG_C for
HOLD_S consecutive seconds it runs ~/handtrack/stop.sh (stops the tracker and
ptgrab), logs why, and exits.  It never restarts anything.  Valve's processes
and the OS's own thermal management are untouched.

Thresholds are conservative engineering limits, not validated safe limits.
"""
import glob
import os
import subprocess
import sys
import time

WATCHDOG_C = 93.0   # user-set 2026-10-11 (was 89); reboot was seen at ~95-97 C
HOLD_S = 3
LOG = os.path.expanduser("~/handtrack/CHANGES.log")


def zones():
    out = []
    for z in glob.glob("/sys/class/thermal/thermal_zone*"):
        try:
            if open(z + "/type").read().strip().startswith(("cpu", "gpuss")):
                out.append(z)
        except OSError:
            pass
    return out


def hottest(zs):
    best, name = 0.0, "?"
    for z in zs:
        try:
            t = int(open(z + "/temp").read()) / 1000.0
        except (OSError, ValueError):
            continue
        if t > best:
            best, name = t, open(z + "/type").read().strip()
    return best, name


def main():
    zs = zones()
    print("watchdog: %d zones, stop at >= %.0f C for %d s" % (len(zs), WATCHDOG_C, HOLD_S), flush=True)
    over = 0
    while True:
        t, name = hottest(zs)
        over = over + 1 if t >= WATCHDOG_C else 0
        if over >= HOLD_S:
            msg = "%s WATCHDOG: %s at %.1f C for %d s -> stopping hand tracking" % (
                time.strftime("%Y-%m-%dT%H:%M:%S"), name, t, HOLD_S)
            print(msg, flush=True)
            with open(LOG, "a") as f:
                f.write(msg + "\n")
            subprocess.run([os.path.expanduser("~/handtrack/stop.sh"), "--from-watchdog"])
            return 0
        time.sleep(1.0)


if __name__ == "__main__":
    sys.exit(main())
