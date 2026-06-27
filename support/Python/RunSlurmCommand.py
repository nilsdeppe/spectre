# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Sequence

from spectre.support.Machines import this_machine

logger = logging.getLogger(__name__)


def in_container() -> bool:
    """True if running inside a spectre Apptainer/Singularity container.

    The container sets the 'SPECTRE_CONTAINER' environment variable (see
    'containers/Dockerfile.buildenv', which sets it to "0", the bash-true
    value); it is unset outside the container. We therefore detect the
    container by the mere presence of the variable. SLURM binaries (such as
    'sbatch' and 'sacct') are not available inside the container, so SLURM
    commands must be routed to the host (see `run_slurm_command`).
    """
    return os.environ.get("SPECTRE_CONTAINER") is not None


def run_slurm_command(
    command: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Run a SLURM command, routing through SSH to the host inside a container.

    Outside a container this is just
    `subprocess.run(command, cwd=cwd, **kwargs)`. Inside a container
    ('SPECTRE_CONTAINER' set, see `in_container`) the SLURM binaries live on
    the host rather than inside the container, so the command is wrapped in an
    SSH call to the host (the container shares the host filesystem via bind
    mounts). The host defaults to 'localhost' but can be overridden with the
    'SPECTRE_CONTAINER_SSH_HOST' environment variable. The SSH private key
    (identity file) can be selected with the 'SPECTRE_CONTAINER_SSH_KEY'
    environment variable or the machine's 'ContainerSshKey' attribute (see
    `_ssh_identity_file`); this is needed on HPC systems where the
    cluster-local key has a non-default name.

    If the SSH call fails because no SSH key is available inside the
    container, a message is logged explaining that the host '$HOME/.ssh'
    directory likely needs to be bind-mounted into the container. The failing
    `subprocess.CompletedProcess` is still returned so callers handle the
    error as usual.

    Arguments:
      command: The SLURM command to run, e.g. ["sbatch", "Submit.sh"].
      cwd: Directory to run the command in. Applied on the host (via 'cd') when
        running through SSH.
      kwargs: Forwarded to `subprocess.run`. Pass 'capture_output=True' and
        'text=True' to enable detection of SSH key failures.
    """
    if not in_container():
        return subprocess.run(command, cwd=cwd, **kwargs)

    host = os.environ.get("SPECTRE_CONTAINER_SSH_HOST", "localhost")
    remote_command = ""
    if cwd is not None:
        remote_command += f"cd {shlex.quote(str(cwd))} && "
    remote_command += " ".join(shlex.quote(str(arg)) for arg in command)
    ssh_command = [
        "ssh",
        # Never prompt for a password, so the call fails fast (exit code 255)
        # instead of hanging when no SSH key is available.
        "-o",
        "BatchMode=yes",
        # Avoid an interactive host-key prompt on the first connection.
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    identity_file = _ssh_identity_file()
    if identity_file is not None:
        ssh_command += ["-i", identity_file]
    ssh_command += [host, remote_command]
    logger.debug(f"Running SLURM command via SSH: {shlex.join(ssh_command)}")
    # 'cwd' is applied on the host via 'cd' above, so don't pass it to the local
    # SSH process.
    result = subprocess.run(ssh_command, **kwargs)
    _warn_if_ssh_key_missing(result, host)
    return result


def _ssh_identity_file() -> Optional[str]:
    """The SSH private key (identity file) to use for the host connection.

    On HPC systems the cluster-local key often has a non-default name (e.g.
    '~/.ssh/cluster') that 'ssh' does not offer automatically, so we pass it
    explicitly with '-i'. The key is resolved from the
    'SPECTRE_CONTAINER_SSH_KEY' environment variable, falling back to the
    'ContainerSshKey' attribute of the configured machine (see
    'support/Machines'). '~' is expanded to the user's home directory. Returns
    'None' if no key is configured, in which case 'ssh' uses its defaults.
    """
    key = os.environ.get("SPECTRE_CONTAINER_SSH_KEY")
    if key is None:
        machine = this_machine(raise_exception=False)
        if machine is not None:
            key = getattr(machine, "ContainerSshKey", None)
    if key:
        return os.path.expanduser(key)
    return None


def _warn_if_ssh_key_missing(
    result: subprocess.CompletedProcess, host: str
) -> None:
    # SSH exits with code 255 on connection or authentication failures. Detect
    # the missing-key case so we can give a targeted hint about binding
    # '$HOME/.ssh' into the container.
    stderr = result.stderr
    if (
        result.returncode == 255
        and isinstance(stderr, str)
        and (
            "Permission denied" in stderr
            or "publickey" in stderr
            or "Host key verification failed" in stderr
        )
    ):
        logger.error(
            f"SSH to the host '{host}' failed, likely because no SSH key was"
            " found inside the container. Most HPC systems use cluster-local"
            " keys in '$HOME/.ssh'. Make sure to bind-mount your host"
            " '$HOME/.ssh' directory into the container (e.g. 'apptainer run"
            " --bind $HOME/.ssh:$HOME/.ssh ...') so the spectre CLI can reach"
            " SLURM on the host. If the cluster-local key has a non-default"
            " name (e.g. '~/.ssh/cluster'), point to it with the"
            " 'SPECTRE_CONTAINER_SSH_KEY' environment variable or the"
            " machine's 'ContainerSshKey' attribute in"
            f" 'support/Machines'.\nUnderlying SSH error:\n{stderr.strip()}"
        )
