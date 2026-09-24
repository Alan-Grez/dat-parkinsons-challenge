from __future__ import annotations

import json
import os
import pickle
import random
import shutil
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import torch


def replace_with_retry(temporary: Path, destination: Path, attempts: int = 12) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            os.replace(temporary, destination)
            return
        except OSError as error:
            last_error = error
            time.sleep(min(0.08 * (2**attempt), 2.0))
    raise RuntimeError(f"No se pudo reemplazar atomicamente {destination}") from last_error


def atomic_json(payload: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    replace_with_retry(temporary, destination)


def atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(temporary, index=False)
    replace_with_retry(temporary, destination)


def atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, temporary)
    replace_with_retry(temporary, destination)


def copy_with_retry(source: Path, destination: Path, attempts: int = 12) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            shutil.copy2(source, temporary)
            replace_with_retry(temporary, destination, attempts=attempts)
            return
        except OSError as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            time.sleep(min(0.08 * (2**attempt), 2.0))
    raise RuntimeError(f"No se pudo copiar de forma segura {source}") from last_error


def torch_load_with_retry(
    path: Path,
    *,
    map_location: str | torch.device,
    attempts: int = 6,
) -> dict[str, Any]:
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            payload = torch.load(path, map_location=map_location, weights_only=False)
            if not isinstance(payload, dict):
                raise TypeError(f"El checkpoint {path} no contiene un diccionario.")
            return payload
        except (OSError, EOFError, RuntimeError, TypeError, pickle.UnpicklingError) as error:
            last_error = error
            time.sleep(min(0.08 * (2**attempt), 1.0))
    raise RuntimeError(f"No se pudo leer el checkpoint {path}") from last_error


def atomic_npz(destination: Path, **arrays: np.ndarray) -> None:
    temporary = destination.with_name(f"{destination.stem}.tmp.{os.getpid()}.npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(temporary, **arrays)
    replace_with_retry(temporary, destination)


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class RunLock(AbstractContextManager["RunLock"]):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.acquired = False

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                previous = json.loads(self.path.read_text(encoding="utf-8"))
                previous_pid = int(previous.get("pid", -1))
            except (ValueError, OSError, json.JSONDecodeError):
                previous_pid = -1
            if process_is_alive(previous_pid):
                raise RuntimeError(
                    f"Ya existe una ejecucion activa (PID {previous_pid}) en {self.path}."
                )
            stale = self.path.with_name(
                f"{self.path.name}.stale.{int(time.time())}.{random.randrange(1000, 9999)}"
            )
            os.replace(self.path, stale)
        descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "created_at": time.time()}, stream)
        self.acquired = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.acquired:
            try:
                self.path.unlink(missing_ok=True)
            finally:
                self.acquired = False


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _cpu_byte_rng_state(value: Any, name: str) -> torch.Tensor:
    """Normalize RNG payloads loaded through any torch ``map_location``."""
    if isinstance(value, torch.Tensor):
        result = value.detach().to(device="cpu", dtype=torch.uint8)
    else:
        try:
            result = torch.as_tensor(value, dtype=torch.uint8, device="cpu")
        except (TypeError, ValueError) as error:
            raise TypeError(f"Estado RNG invalido para {name}.") from error
    return result.contiguous().reshape(-1)


def restore_rng_state(state: dict[str, Any]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # Loading a checkpoint with map_location="cuda" also moves this CPU RNG
    # tensor to CUDA. torch.set_rng_state strictly requires a CPU ByteTensor.
    torch.set_rng_state(_cpu_byte_rng_state(state["torch_cpu"], "torch_cpu"))
    if torch.cuda.is_available() and "torch_cuda" in state:
        cuda_states = [
            _cpu_byte_rng_state(value, f"torch_cuda[{index}]")
            for index, value in enumerate(state["torch_cuda"])
        ]
        torch.cuda.set_rng_state_all(cuda_states)
