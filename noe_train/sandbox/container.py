"""Docker container lifecycle management for episode sandboxes."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass()
class ContainerConfig:
    image: str
    memory_limit: str = "4g"
    cpu_count: int = 2
    timeout: int = 300  # episode timeout in seconds
    network_mode: str = "none"  # no network access in sandbox


class SandboxContainer:
    """Manages a Docker container for a single episode."""

    def __init__(self, config: ContainerConfig):
        import docker

        self.config = config
        self.client = docker.from_env()
        self._container = None

    @property
    def container_id(self) -> str | None:
        return self._container.id if self._container else None

    def start(self, episode_id: str, env_vars: dict[str, str] | None = None) -> str:
        """Start a new sandbox container. Returns container ID."""
        labels = {"noe.episode_id": episode_id}
        environment = env_vars or {}
        environment["EPISODE_ID"] = episode_id

        self._container = self.client.containers.run(
            self.config.image,
            detach=True,
            labels=labels,
            mem_limit=self.config.memory_limit,
            nano_cpus=self.config.cpu_count * 1_000_000_000,
            network_mode=self.config.network_mode,
            environment=environment,
            stdin_open=True,
            tty=False,
        )
        logger.info(f"Started container {self._container.id[:12]} for episode {episode_id}")
        return self._container.id

    def exec_command(self, cmd: str, timeout: int = 60) -> tuple[int, str]:
        """Execute a command in the container. Returns (exit_code, output)."""
        if self._container is None:
            raise RuntimeError("Container not started")
        try:
            exit_code, output = self._container.exec_run(
                cmd,
                demux=False,
                timeout=timeout,
            )
            text = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else str(output)
            return exit_code, text
        except Exception as e:
            return -1, f"exec error: {e}"

    def copy_to(self, local_path: str, container_path: str) -> bool:
        """Copy a file into the container."""
        if self._container is None:
            raise RuntimeError("Container not started")
        import io
        import tarfile

        with open(local_path, "rb") as f:
            data = f.read()

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            info = tarfile.TarInfo(name=container_path.split("/")[-1])
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        tar_stream.seek(0)

        dest_dir = "/".join(container_path.split("/")[:-1]) or "/"
        return self._container.put_archive(dest_dir, tar_stream)

    def stop(self) -> None:
        """Stop and remove the container."""
        if self._container is not None:
            try:
                self._container.stop(timeout=10)
                self._container.remove(force=True)
                logger.info(f"Stopped container {self._container.id[:12]}")
            except Exception as e:
                logger.warning(f"Error stopping container: {e}")
            finally:
                self._container = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
