#!/usr/bin/env python3
"""Host-wide cooperative reservation for this machine's single Ascend NPU.

Only the Python standard library is required. See docs/ascend950/development.md.
"""

import argparse
import contextlib
import ctypes as C
import datetime as dt
import fcntl
import json
import math
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

LOCK_DIR = Path("/run/lock/ascend-npu")
LOCK_NAME = "device-0.lock"
DRIVER = "/usr/local/Ascend/driver/lib64/driver/libdrvdsmi_host.so"
SCRIPT = str(Path(__file__).resolve())
PROBE_SCRIPTS = {SCRIPT, "/usr/local/bin/npu-guard"}
BUSY = 75
CONFLICT = 76
UNAVAILABLE = 78


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def proc_identity(pid):
    """PID start ticks prevent confusing a recycled PID with an old owner."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "start_ticks": int(fields[19]),
            "uid": Path(f"/proc/{pid}").stat().st_uid,
        }
    except (OSError, ValueError, IndexError):
        return {"pid": pid, "start_ticks": None, "uid": None}


class ResourcePara(C.Structure):
    _fields_ = [
        ("owner_type", C.c_uint),
        ("owner_id", C.c_uint),
        ("resource_type", C.c_uint),
        ("tsid", C.c_uint),
        ("reserve", C.c_uint * 8),
    ]


class ResourceInfo(C.Structure):
    _fields_ = [("buf_len", C.c_uint), ("buf", C.c_void_p), ("reserve", C.c_uint * 8)]


class MemoryInfo(C.Structure):
    _fields_ = [("total_kib", C.c_uint), ("used_kib", C.c_uint), ("reserve", C.c_char * 38)]


class CoreFrequency(C.Structure):
    _fields_ = [("rated_mhz", C.c_uint), ("current_mhz", C.c_uint)]


def readonly_guard_probe(entry):
    """Known stdlib-only observers do not create compute contexts.

    This is cooperative classification, not an authorization boundary. An
    uncooperative user can bypass flock altogether. Unknown tools stay candidates.
    """
    try:
        args = (entry / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        return (
            len(args) == 3
            and args[2] == b"_probe"
            and os.path.realpath(os.fsdecode(args[1])) in PROBE_SCRIPTS
        )
    except (OSError, ValueError):
        return False


def scan_device_fds(proc=Path("/proc")):
    """Conservatively include runtime initialization, not just /dev/davinci0.

    A manager/HDC FD is a candidate, not proof of compute activity. An
    incomplete host view must never be interpreted as an empty device.
    """
    candidates, unreadable = [], []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            nodes = set()
            for fd in (entry / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except FileNotFoundError:
                    continue
                if target.startswith(("/dev/davinci", "/dev/devmm", "/dev/hisi_hdc")):
                    nodes.add(target)
            if nodes and not readonly_guard_probe(entry):
                candidates.append({**proc_identity(int(entry.name)), "nodes": sorted(nodes)})
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            unreadable.append(int(entry.name))
    return candidates, unreadable


def classify(sample):
    reasons = []
    if sample.get("pids"):
        reasons.append("driver reports NPU processes")
    if sample.get("fd_candidates"):
        reasons.append("other processes hold Ascend device FDs (possibly initializing)")
    if any(
        v.get("rc") == 0 and 0 < v.get("value", 0) <= 100
        for v in sample.get("utilization", {}).values()
    ):
        reasons.append("driver reports positive utilization")
    if reasons:
        return "busy", reasons
    errors = list(sample.get("errors", []))
    if sample.get("unreadable_proc_pids"):
        errors.append(
            "incomplete /proc FD visibility; use a host account with sufficient permissions"
        )
    if sample.get("health") != {"rc": 0, "value": 0}:
        errors.append("device health is unavailable or not healthy")
    if sample.get("pid_query_rc") != 0:
        errors.append("driver process query failed")
    if errors:
        return "unknown", errors
    return "idle", ["no driver processes or Ascend FD candidates in this sample"]


def raw_probe():
    """Read-only DSMI calls in a disposable process, using installed ABI headers.

    dsmi_common_interface.h: PROCESS_RESOURCE=2, DEV_PROCESS_PID=8.
    Ascend950 asys handler: get_device_info(MEMORY=9, HBM_MEMORY=35).
    Never initialize torch/ACL, submit work, reset, or alter device settings.
    """
    result = {"time": now(), "device": 0, "errors": [], "utilization": {}}
    try:
        devices = sorted(str(p) for p in Path("/dev").glob("davinci[0-9]*"))
        if devices != ["/dev/davinci0"]:
            result["errors"].append(f"expected only /dev/davinci0; found {devices}")
        lib = C.CDLL(DRIVER)
        lib.dsmi_get_resource_info.argtypes = [
            C.c_uint,
            C.POINTER(ResourcePara),
            C.POINTER(ResourceInfo),
        ]
        lib.dsmi_get_device_health.argtypes = [C.c_int, C.POINTER(C.c_uint)]
        lib.dsmi_get_device_utilization_rate.argtypes = [C.c_int, C.c_int, C.POINTER(C.c_uint)]
        lib.dsmi_get_device_info.argtypes = [
            C.c_int,
            C.c_int,
            C.c_int,
            C.c_void_p,
            C.POINTER(C.c_uint),
        ]
        lib.dsmi_get_aicore_info.argtypes = [C.c_int, C.POINTER(CoreFrequency)]
        lib.dsmi_get_device_power_info.argtypes = [C.c_int, C.POINTER(C.c_ushort)]
        pids = (C.c_int * 64)()
        para = ResourcePara(2, 0, 8, 0)
        info = ResourceInfo(C.sizeof(pids), C.cast(pids, C.c_void_p))
        rc = lib.dsmi_get_resource_info(0, C.byref(para), C.byref(info))
        result["pid_query_rc"] = rc
        result["pids"] = [proc_identity(pid) for pid in pids if pid > 0] if rc == 0 else []
        if rc == 0 and (info.buf_len > C.sizeof(pids) or any(pid < 0 for pid in pids)):
            result["errors"].append("invalid driver process response")
        if len(result["pids"]) == 64:
            result["errors"].append("process list may be truncated")
        health = C.c_uint()
        rc = lib.dsmi_get_device_health(0, C.byref(health))
        result["health"] = {"rc": rc, "value": health.value}
        for kind, name in [
            (2, "aicore"),
            (10, "hbm_bw"),
            (12, "vector"),
            (13, "npu"),
            (14, "cube"),
        ]:
            value = C.c_uint()  # Some current driver paths write only the low byte.
            rc = lib.dsmi_get_device_utilization_rate(0, kind, C.byref(value))
            result["utilization"][name] = {"rc": rc, "value": value.value}
            if rc == 0 and value.value > 100:
                result["errors"].append(f"invalid {name} utilization")
        if result["utilization"]["npu"]["rc"] != 0:
            result["errors"].append("NPU overall utilization query failed")
        mem = MemoryInfo()
        size = C.c_uint(C.sizeof(mem))
        rc = lib.dsmi_get_device_info(0, 9, 35, C.byref(mem), C.byref(size))
        result["memory"] = {
            "rc": rc,
            "total_mib": mem.total_kib / 1024,
            "used_mib": mem.used_kib / 1024,
        }
        result["core_frequency"] = []
        for component in (0, 1):
            freq = CoreFrequency()
            rc = lib.dsmi_get_aicore_info(component << 16, C.byref(freq))
            result["core_frequency"].append({"rc": rc, "mhz": freq.current_mhz})
        power = C.c_ushort()
        rc = lib.dsmi_get_device_power_info(0, C.byref(power))
        result["power"] = {"rc": rc, "watts": power.value / 10}
    except (OSError, AttributeError) as exc:
        result["errors"].append(str(exc))
    result["fd_candidates"], result["unreadable_proc_pids"] = scan_device_fds()
    result["state"], result["reasons"] = classify(result)
    return result


def probe():
    try:
        completed = subprocess.run(
            [sys.executable, SCRIPT, "_probe"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        # Some driver builds print diagnostic messages on stdout, including
        # during library teardown. Keep our structured record separate.
        for line in reversed(completed.stdout.splitlines()):
            try:
                sample = json.loads(line)
            except ValueError:
                continue
            if (
                isinstance(sample, dict)
                and sample.get("state") in ("idle", "busy", "unknown")
                and "reasons" in sample
            ):
                return sample
        raise ValueError("probe did not produce a structured status record")
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        return {"time": now(), "state": "unknown", "reasons": [f"probe failed: {exc}"]}


def open_lock():
    """Root creates a persistent inode. Clients never unlink or replace it."""
    if not LOCK_DIR.exists():
        if os.geteuid() != 0:
            raise PermissionError("root must run npu_guard.py init once after boot")
        with contextlib.suppress(FileExistsError):
            LOCK_DIR.mkdir(mode=0o755)
    directory = LOCK_DIR.lstat()
    if not stat.S_ISDIR(directory.st_mode) or directory.st_uid != 0 or directory.st_mode & 0o022:
        raise PermissionError(f"{LOCK_DIR} must be a root-owned, non-writable real directory")
    path = LOCK_DIR / LOCK_NAME
    if os.geteuid() == 0:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o666)
        except FileExistsError:
            pass
        else:
            os.fchmod(fd, 0o666)
            os.close(fd)
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or st.st_uid != 0 or st.st_nlink != 1:
        os.close(fd)
        raise PermissionError("lock must be a root-owned regular file with one link")
    return fd


def try_lock(fd):
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def read_metadata(fd):
    try:
        return json.loads(os.pread(fd, 65536, 0))
    except (ValueError, OSError):
        return None


def write_metadata(fd, data):
    data["updated_at"] = now()
    encoded = (json.dumps(data, ensure_ascii=False) + "\n").encode()
    os.pwrite(fd, encoded, 0)
    os.ftruncate(fd, len(encoded))


def emit(message):
    print(f"[npu-guard] {message}", file=sys.stderr, flush=True)


def group_alive(pgid):
    # Zombies cannot use the NPU and must not prevent releasing a reservation.
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != "Z":
                return True
        except (FileNotFoundError, ProcessLookupError):
            pass
    return False


def foreign_processes(sample, pgid):
    """Only this invocation's process group owns its reservation."""
    foreign = set()
    # Runtime monitoring uses registered contexts. Read-only status probes can
    # briefly open manager FDs and must not abort a running workload.
    for item in sample.get("pids", []):
        pid = item["pid"]
        try:
            if os.getpgid(pid) != pgid:
                foreign.add(pid)
        except ProcessLookupError:
            continue
    return sorted(foreign)


def execute(fd, metadata, command, timeout, grace=5, sample_fn=probe):
    """Keep the reservation through foreground work and descendants.

    The child inherits the same flock FD: killing only this supervisor does
    not release the reservation while the immediate child is still alive.
    Surviving runtime processes are also caught by the next physical probe.
    """
    received = []
    old_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        old_handlers[sig] = signal.signal(sig, lambda number, frame: received.append(number))
    child = None
    try:
        env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES="0", ASCEND_VISIBLE_DEVICES="0")
        # A fresh session permits terminating only this invocation's process group.
        child = subprocess.Popen(command, pass_fds=(fd,), start_new_session=True, env=env)
        metadata.update(phase="running", child=proc_identity(child.pid))
        write_metadata(fd, metadata)
        started, stopping, stop_code = time.monotonic(), None, None
        next_probe = started + 2
        while True:
            code = child.poll()
            alive = group_alive(child.pid)
            if code is not None and not alive:
                return stop_code if stop_code is not None else (code if code >= 0 else 128 - code)
            timed_out = timeout is not None and time.monotonic() - started >= timeout
            conflict_code = None
            if (
                stopping is None
                and not received
                and not timed_out
                and time.monotonic() >= next_probe
            ):
                sample = sample_fn()
                foreign = foreign_processes(sample, child.pid)
                if foreign:
                    emit(f"external Ascend users appeared: {foreign}; stopping this workload")
                    metadata["external_pids"] = foreign
                    conflict_code = CONFLICT
                elif (
                    sample["state"] == "unknown"
                    or sample.get("errors")
                    or sample.get("unreadable_proc_pids")
                    or sample.get("pid_query_rc", 0) != 0
                    or sample.get("health", {"rc": 0, "value": 0}) != {"rc": 0, "value": 0}
                ):
                    emit("device visibility lost; stopping this workload")
                    conflict_code = UNAVAILABLE
                next_probe = time.monotonic() + 2
            if stopping is None and (received or timed_out or conflict_code):
                sig = received[0] if received else signal.SIGTERM
                stop_code = 128 + sig if received else (conflict_code or 124)
                stopping = time.monotonic()
                metadata["phase"] = "stopping"
                write_metadata(fd, metadata)
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, sig)
            elif stopping is not None and time.monotonic() - stopping >= grace:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
            time.sleep(0.1)
    except BaseException:
        # Unexpected observer/logging failures must not leave an unmonitored
        # command behind. Its session/group was created by this invocation.
        if child is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                child.wait(timeout=5)
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def run(args, sample_fn=probe):
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("provide a foreground command after --")
    fd = open_lock()
    acquired = False
    metadata = {
        **proc_identity(os.getpid()),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "owner": args.owner,
        "device": 0,
        "command": Path(command[0]).name,
        "phase": "waiting_lock",
        "requested_at": now(),
    }
    deadline = time.monotonic() + args.wait
    last_message, last_report, quiet_since = None, 0, None
    try:
        while True:
            if not acquired:
                acquired = try_lock(fd)
                if acquired:
                    metadata.update(phase="waiting_device", acquired_at=now())
                    write_metadata(fd, metadata)
            if acquired:
                sample = sample_fn()
                state = sample["state"]
                message = f"device {state}: {'; '.join(sample['reasons'])}"
                if state == "unknown":
                    emit(message)
                    return UNAVAILABLE
                if state == "idle":
                    if quiet_since is None:
                        quiet_since = time.monotonic()
                    if time.monotonic() - quiet_since >= args.idle_seconds:
                        emit(f"acquired device 0 for {args.owner}")
                        result = execute(fd, metadata, command, args.timeout, sample_fn=sample_fn)
                        metadata.update(phase="finished", exit_code=result)
                        return result
                else:
                    quiet_since = None
                    message += f"; pids={[p['pid'] for p in sample.get('pids', [])]}"
            else:
                holder = read_metadata(fd)
                message = f"waiting for reservation: {holder}"
            current = time.monotonic()
            if message != last_message or current - last_report >= 30:
                emit(message)
                last_message, last_report = message, current
            if current >= deadline:
                emit("wait deadline reached; no workload launched")
                return BUSY
            time.sleep(min(args.poll, max(0, deadline - current)))
    finally:
        try:
            if acquired:
                if metadata["phase"] != "finished":
                    metadata["phase"] = (
                        "aborted" if "child" in metadata else "released_without_launch"
                    )
                write_metadata(fd, metadata)
        finally:
            # Never LOCK_UN: a surviving child may share this open file description.
            os.close(fd)


def status(as_json):
    sample = probe()
    try:
        fd = open_lock()
    except OSError as exc:
        sample["reservation"] = {"error": str(exc)}
        sample["state"] = "unknown"
        sample["reasons"].append("shared lock is unavailable")
    else:
        try:
            free = try_lock(fd)
            sample["reservation"] = {"locked": not free, "metadata": read_metadata(fd)}
            if not free:
                sample["state"] = "busy"
                sample["reasons"].append("cooperative reservation is held")
        finally:
            os.close(fd)
    if as_json:
        print(json.dumps(sample, ensure_ascii=False))
    else:
        print(f"device 0: {sample['state']}")
        print(json.dumps(sample, ensure_ascii=False, indent=2))
    return {"idle": 0, "busy": BUSY, "unknown": UNAVAILABLE}[sample["state"]]


def finite_number(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    subs.add_parser("init", help="root: initialize the host lock after boot")
    state = subs.add_parser("status", help="read-only device and reservation snapshot")
    state.add_argument("--json", action="store_true")
    job = subs.add_parser(
        "run", help="reserve, wait for physical availability, run foreground command"
    )
    job.add_argument("--owner", required=True, help="agent/task name (no secrets)")
    job.add_argument(
        "--wait",
        type=finite_number,
        default=600,
        help="acquisition deadline in seconds (default 600)",
    )
    job.add_argument("--poll", type=finite_number, default=1)
    job.add_argument(
        "--idle-seconds", type=finite_number, default=3, help="continuous quiet window (default 3)"
    )
    job.add_argument(
        "--timeout", type=finite_number, help="workload wall time; kills only its process group"
    )
    job.add_argument("command", nargs=argparse.REMAINDER)
    subs.add_parser("_probe", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.action == "_probe":
            print(json.dumps(raw_probe(), ensure_ascii=False))
            return 0
        if args.action == "init":
            fd = open_lock()
            os.close(fd)
            print(LOCK_DIR / LOCK_NAME)
            return 0
        if args.action == "status":
            return status(args.json)
        if args.poll <= 0:
            parser.error("--poll must be positive")
        return run(args)
    except (OSError, ValueError) as exc:
        emit(str(exc))
        return UNAVAILABLE
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
