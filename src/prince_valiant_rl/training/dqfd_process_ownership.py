"""Lightweight PID/create-time ancestry validation for the supervised DQfD child."""
from __future__ import annotations

import os
from typing import Any, Callable


def process_record(pid: int) -> dict[str, Any]:
    """Read only the identity fields needed for the bounded ownership chain."""
    import psutil
    process = psutil.Process(int(pid))
    with process.oneshot():
        return {"pid": int(process.pid), "ppid": int(process.ppid()),
                "create_time": float(process.create_time()), "exe": str(process.exe())}


def validate_private_child_ownership(*, child_pid: int, parent_pid: int,
        parent_create_time: float, parent_exe: str, launcher_exe: str,
        child_exe: str, process_inspector: Callable[[int], dict[str, Any]] = process_record
        ) -> dict[str, Any]:
    """Validate direct or one Windows venv-launcher hop by PID, creation time and executable."""
    if not all((int(child_pid) > 0, int(parent_pid) > 0, float(parent_create_time) > 0,
                parent_exe, launcher_exe, child_exe)):
        raise PermissionError("private child ownership identity is incomplete")
    try:
        child = process_inspector(int(child_pid))
        parent = process_inspector(int(parent_pid))
        if child["pid"] != int(child_pid) or parent["pid"] != int(parent_pid):
            raise PermissionError("private child ownership PID identity changed")
        if float(parent["create_time"]) != float(parent_create_time) or os.path.normcase(parent["exe"]) != os.path.normcase(parent_exe):
            raise PermissionError("private child declared parent identity drifted")
        observed = [child]
        if int(child["ppid"]) == int(parent_pid):
            if os.path.normcase(child["exe"]) != os.path.normcase(child_exe):
                raise PermissionError("private child executable differs from the declared interpreter")
        else:
            launcher = process_inspector(int(child["ppid"]))
            observed.append(launcher)
            if (int(launcher["ppid"]) != int(parent_pid)
                    or os.path.normcase(launcher["exe"]) != os.path.normcase(launcher_exe)
                    or os.path.normcase(child["exe"]) != os.path.normcase(child_exe)):
                raise PermissionError("private child ancestry is not the declared one-hop interpreter launcher")
        return {"status": "validated", "declared_parent": {"pid": int(parent_pid),
            "create_time": float(parent_create_time), "exe": str(parent_exe)},
            "observed_ancestry": [{"pid": int(row["pid"]), "ppid": int(row["ppid"]),
                "create_time": float(row["create_time"]), "exe": str(row["exe"])} for row in observed]}
    except PermissionError:
        raise
    except Exception as exc:
        raise PermissionError("private child process ownership evidence unavailable") from exc
