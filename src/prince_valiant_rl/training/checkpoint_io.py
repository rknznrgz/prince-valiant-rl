"""Atomic temporal checkpoint writes with conservative endpoint deduplication."""
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


class CheckpointFinalizationError(OSError):
    """An atomic checkpoint replacement failed; the prior destination is intact."""

    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        destination_state = diagnostics.get('destination_state', {})
        pending_state = diagnostics.get('pending_state', {})
        super().__init__(
            f"Checkpoint replacement failed: {diagnostics['pending']} -> "
            f"{diagnostics['destination']}; destination_preserved="
            f"{diagnostics['destination_exists']}; destination_state={destination_state}; "
            f"pending_state={pending_state}; error={diagnostics['error']}"
        )


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _path_diagnostics(path):
    path = Path(path)
    row = dict(path=str(path), exists=path.is_file())
    if row['exists']:
        try:
            row['size_bytes'] = path.stat().st_size
            row['sha256'] = _file_sha256(path)
        except OSError as exc:
            row['diagnostic_error'] = repr(exc)
    return row


def _feed(digest, tag, value):
    digest.update(tag.encode('utf-8'))
    digest.update(len(value).to_bytes(8, 'big'))
    digest.update(value)


def checkpoint_state_sha256(payload):
    """Hash every saved field except volatile wall-clock metadata.

    ``metadata.wall_seconds`` can advance after the final cadence save without
    changing any state needed to resume. All model, optimizer, RNG, curriculum,
    worker snapshot, case, counter, configuration, and other metadata fields
    remain part of the comparison.
    """
    digest = hashlib.sha256()

    def visit(value, path=()):
        if path == ('metadata', 'wall_seconds'):
            _feed(digest, 'ignored-wall-seconds', b'')
        elif isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            _feed(digest, 'torch.dtype', str(tensor.dtype).encode())
            _feed(digest, 'torch.shape', json.dumps(list(tensor.shape)).encode())
            _feed(digest, 'torch.bytes', tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, np.ndarray):
            _feed(digest, 'numpy.dtype', value.dtype.str.encode())
            _feed(digest, 'numpy.shape', json.dumps(list(value.shape)).encode())
            _feed(digest, 'numpy.bytes', np.ascontiguousarray(value).tobytes())
        elif isinstance(value, np.generic):
            visit(value.item(), path)
        elif isinstance(value, Mapping):
            _feed(digest, 'mapping.type', type(value).__qualname__.encode())
            keys = sorted(value.keys(), key=lambda key: (type(key).__qualname__, repr(key)))
            _feed(digest, 'mapping.length', str(len(keys)).encode())
            for key in keys:
                visit(key, path + ('<key>',))
                visit(value[key], path + ((key if isinstance(key, str) else repr(key)),))
        elif isinstance(value, (list, tuple)):
            _feed(digest, 'sequence.type', type(value).__qualname__.encode())
            _feed(digest, 'sequence.length', str(len(value)).encode())
            for index, item in enumerate(value):
                visit(item, path + (str(index),))
        elif value is None or isinstance(value, (str, bytes, int, float, bool)):
            _feed(digest, type(value).__qualname__, repr(value).encode('utf-8'))
        else:
            raise TypeError(f'Unsupported checkpoint identity value: {type(value).__qualname__}')

    visit(payload)
    return digest.hexdigest()


def _saved_state_matches(path, fingerprint):
    try:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        return checkpoint_state_sha256(payload) == fingerprint
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, EOFError):
        return False


def atomic_torch_save(payload, path, *, skip_if_identical=False):
    """Write to a sibling pending file and atomically replace the destination.

    The destination is never unlinked before replacement. On replacement error,
    both the prior destination and pending file are retained for diagnosis.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = None
    try:
        fingerprint = checkpoint_state_sha256(payload)
    except Exception:
        # Any fingerprinting failure only disables deduplication; serialization
        # and the atomic replacement contract remain available. In particular,
        # never skip a save unless the complete state was compared successfully.
        pass
    if (skip_if_identical and fingerprint is not None and path.is_file()
            and _saved_state_matches(path, fingerprint)):
        return dict(saved=False, identical_state=True, state_sha256=fingerprint,
                    destination=str(path), pending=None)

    temporary = path.with_suffix('.pending')
    torch.save(payload, temporary)
    try:
        temporary.replace(path)
    except OSError as exc:
        diagnostics = dict(destination=str(path), pending=str(temporary),
            destination_exists=path.is_file(), pending_exists=temporary.is_file(),
            error=repr(exc), errno=getattr(exc, 'errno', None),
            winerror=getattr(exc, 'winerror', None),
            destination_state=_path_diagnostics(path), pending_state=_path_diagnostics(temporary))
        raise CheckpointFinalizationError(diagnostics) from exc
    return dict(saved=True, identical_state=False, state_sha256=fingerprint,
                destination=str(path), pending=str(temporary))
