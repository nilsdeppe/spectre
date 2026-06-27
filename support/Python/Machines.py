# Distributed under the MIT License.
# See LICENSE.txt for details.
"""Support for host machines, such as supercomputers.

Machines are defined as YAML files in 'support/Machines/'. To add support for a
new machine, add a YAML file that defines a `Machine:` key with the attributes
listed in the `Machine` class below. Also add a submit script with the same
name to 'support/SubmitScripts/'.

To select a machine, specify the `MACHINE` option when configuring the CMake
build.
"""

import glob
import os
import re
import socket
import threading
from dataclasses import dataclass
from typing import Iterator, List, Optional

# functools.cache was added in Py 3.9. Fall back to 'lru_cache' in earlier
# versions, which is pretty much the same but slightly slower.
try:
    from functools import cache
except ImportError:
    from functools import lru_cache

    cache = lru_cache(maxsize=None)

import yaml


@dataclass(frozen=True)
class Machine(yaml.YAMLObject):
    """A machine we know how to run on, such as a particular supercomputer.

    Many configuration options for job submission are set in the submit script
    for the machine (in 'support/SubmitScripts/'). Here we provide additional
    metadata about the machine.

    Attributes:
      Name: A short name for the machine. Must match the YAML file name.
      Description: A description of the machine. Give some basic context and
        any information that may help people get started using the machine.
        Provide links to wiki pages, signup pages, etc., for additional
        information.
      DefaultTasksPerNode: Default number of tasks per node (MPI ranks).
        Often chosen to be the number of sockets on a node.
      DefaultProcsPerTask: Default number of worker threads spawned per task.
        It is often advised to leave one core per node or socket free for
        communication, so this might be the number of cores or hyperthreads
        per node or socket minus one.
      DefaultQueue: Default queue that jobs are submitted to. On Slurm systems
        you can see the available queues with `sinfo`.
      DefaultTimeLimit: Default wall time limit for submitted jobs. For
        acceptable formats, see: https://slurm.schedmd.com/sbatch.html#OPT_time
      LaunchCommandSingleNode: Command to launch an executable on a single
        compute node, e.g. ["mpirun", "-n", "1"]. This is used to run
        executables on interactive compute nodes. This is _not_ the full command
        to launch an executable in scheduled jobs, which can be found in the
        submit script instead. This is also not used on non-compute (login)
        nodes.
      LaunchCommandLoginNode: Command to launch an executable on a the
        login node. This is the counterpart to `LaunchCommandSingleNode` which
        is to be used as a prefix for commands run on non-compute (login) nodes.
      ContainerSshKey: Optional path to the SSH private key (identity file) used
        to reach the host from inside a container. When the spectre CLI runs
        inside a container (see 'SPECTRE_CONTAINER'), SLURM commands are routed
        through 'ssh' to the host; on HPC systems the cluster-local key often
        has a non-default name (e.g. '~/.ssh/cluster') that 'ssh' does not try
        automatically. Set this to that key path. '~' is expanded. The
        'SPECTRE_CONTAINER_SSH_KEY' environment variable overrides this value.
      HostnameRegex: Optional regular expression that matches the hostnames of
        this machine's head and compute nodes. Used to identify the current
        machine from its hostname (see 'this_machine_by_hostname'). Try to make
        this as specific as possible: a regex that also matches a different
        machine is a problem. The pattern is matched with an unanchored search,
        and is written to be compatible with Python ('re'), Perl ('m//'), and
        CMake's regex engine, so it can be shared with SpEC's 'Machines.pm' and
        with 'cmake/SetupMachine.cmake'. For CMake compatibility, use POSIX
        character classes like '[0-9]' and avoid Perl shorthands such as '\\d',
        '\\w', or '\\s'.
      LoginNodeRegex: Optional regular expression that matches the hostnames of
        this machine's login (head) nodes only. Counterpart to 'HostnameRegex'
        for login-node-specific behavior.
    """

    yaml_tag = "!Machine"
    yaml_loader = yaml.SafeLoader
    # The YAML machine files can have these attributes:
    Name: str
    Description: str
    DefaultTasksPerNode: int
    DefaultProcsPerTask: int
    DefaultQueue: str
    DefaultTimeLimit: str
    LaunchCommandSingleNode: List[str]
    LaunchCommandLoginNode: List[str]
    # Optional attributes (may be absent in a machine's YAML file, so access
    # them with 'getattr(machine, ..., default)'):
    ContainerSshKey: Optional[str] = None
    HostnameRegex: Optional[str] = None
    LoginNodeRegex: Optional[str] = None

    def on_compute_node(self) -> bool:
        """Determines whether or not we are running on a compute node."""
        return os.environ.get("SLURM_JOB_ID") is not None

    @property
    def launch_command(self) -> List[str]:
        """The command to launch an executable on the machine.

        Prepend this list to the command you want to run.
        """
        if self.on_compute_node():
            return self.LaunchCommandSingleNode
        else:
            return self.LaunchCommandLoginNode


# Parse YAML machine files as Machine objects
yaml.SafeLoader.add_path_resolver("!Machine", ["Machine"], dict)


class UnknownMachineError(Exception):
    """Indicates we were unsuccessful in identifying the current machine"""

    pass


@cache
def this_machine(
    machinefile_path=os.path.join(os.path.dirname(__file__), "Machine.yaml"),
    raise_exception=True,
    machines_dir=os.path.join(os.path.dirname(__file__), "Machines"),
) -> Machine:
    """Determine the machine we are running on.

    Specify the 'MACHINE' option in the CMake build configuration to select a
    machine, or pass the 'machinefile_path' argument to this function. If no
    machine was selected, we try to identify the machine from its hostname (see
    'this_machine_by_hostname').

    Arguments:
      machinefile_path: Path to a YAML file that describes the current machine.
        Defaults to the machine selected in the CMake build configuration.
      raise_exception: If 'True' (default), raises an 'UnknownMachineError' if
        no machine was selected. Otherwise, returns 'None' if no machine was
        selected.
      machines_dir: Directory of machine YAML files used for the hostname-based
        fallback. Defaults to the 'Machines' directory next to this module.

    Returns: A 'Machine' object that describes the current machine, or 'None' if
      no machine was selected and 'raise_exception' is 'False'.
    """
    if not os.path.exists(machinefile_path):
        # No machine was selected at build time. Try to identify the machine
        # from its hostname before giving up.
        machine = this_machine_by_hostname(
            machines_dir=machines_dir, raise_exception=False
        )
        if machine is not None:
            return machine
        if not raise_exception:
            return None
        # Surface the specific hostname error (e.g. an ambiguous 'HostnameRegex'
        # configuration that matched multiple machines), augmented with the
        # build-time selection hint. Re-running with exceptions enabled is
        # cheap and only happens on this error path.
        try:
            this_machine_by_hostname(machines_dir=machines_dir)
        except UnknownMachineError as error:
            raise UnknownMachineError(
                f"{error}\nAlternatively, no machine was selected at build "
                "time. Specify the 'MACHINE' option when configuring the build "
                "with CMake. The machine file was expected at the following "
                f"path:\n  {machinefile_path}"
            ) from error
    with open(machinefile_path, "r") as open_machinefile:
        return yaml.safe_load(open_machinefile)["Machine"]


def all_machines(
    machines_dir=os.path.join(os.path.dirname(__file__), "Machines"),
) -> List[Machine]:
    """Load all known machines.

    Arguments:
      machines_dir: Directory containing the machine YAML files. Defaults to the
        'Machines' directory next to this module, which is populated by CMake.

    Returns: A list of 'Machine' objects, one per YAML file in 'machines_dir'.
    """
    machines = []
    for machinefile_path in sorted(
        glob.glob(os.path.join(machines_dir, "*.yaml"))
    ):
        with open(machinefile_path, "r") as open_machinefile:
            machines.append(yaml.safe_load(open_machinefile)["Machine"])
    return machines


# How long to wait for the reverse-name lookup in '_fqdn' before giving up, in
# seconds. Resolvers that can answer at all answer much faster than this.
_FQDN_LOOKUP_TIMEOUT = 2.0


@cache
def _fqdn(timeout: float = _FQDN_LOOKUP_TIMEOUT) -> Optional[str]:
    """The fully-qualified hostname, or 'None' if it isn't resolved in time.

    'socket.getfqdn' does a reverse-name lookup, which blocks for as long as the
    resolver takes to give up when the hostname has no record. That is the
    common case on laptops, and in particular on macOS, where the '.local' mDNS
    name is handed to 'mDNSResponder' and the lookup can take minutes. Since
    identifying the machine is a convenience, we run the lookup in a thread and
    give up after 'timeout' seconds rather than stall every CLI invocation. The
    thread is a daemon so an abandoned lookup can't hold up interpreter exit,
    and the result is cached so we wait at most once per process.
    """
    resolved = []

    def lookup() -> None:
        try:
            resolved.append(socket.getfqdn())
        except OSError:
            pass

    thread = threading.Thread(target=lookup, daemon=True)
    thread.start()
    thread.join(timeout)
    return resolved[0] if resolved else None


def _local_hostnames() -> Iterator[str]:
    """Hostnames of the current machine to match against, cheapest first.

    'socket.gethostname' is a syscall that needs no name resolution, so it comes
    first and is all we need for machines whose 'HostnameRegex' doesn't include
    the domain. The fully-qualified name is only resolved if the caller consumes
    the second item, because the reverse lookup it needs can be slow (see
    '_fqdn'). Machines such as 'CaltechHpc' ('login1.cm.cluster') need it.
    """
    hostname = socket.gethostname()
    yield hostname
    fqdn = _fqdn()
    if fqdn is not None and fqdn != hostname:
        yield fqdn


def _machines_matching(hostname: str, machines_dir: str) -> List[Machine]:
    """All known machines whose 'HostnameRegex' matches 'hostname'."""
    return [
        machine
        for machine in all_machines(machines_dir)
        if machine.HostnameRegex is not None
        and re.search(machine.HostnameRegex, hostname) is not None
    ]


def this_machine_by_hostname(
    hostname=None,
    machines_dir=os.path.join(os.path.dirname(__file__), "Machines"),
    raise_exception=True,
) -> Machine:
    """Identify the current machine from its hostname.

    Matches the hostname against the 'HostnameRegex' of every known machine
    (see 'all_machines'). This is independent of the machine selected at build
    time with the 'MACHINE' CMake option.

    Arguments:
      hostname: Hostname to match. Defaults to the hostnames of the current
        machine, tried cheapest first (see '_local_hostnames').
      machines_dir: Directory containing the machine YAML files. Defaults to the
        'Machines' directory next to this module, which is populated by CMake.
      raise_exception: If 'True' (default), raises an 'UnknownMachineError' if
        no machine matches the hostname, or if more than one machine matches
        (an ambiguous 'HostnameRegex' configuration). Otherwise, returns 'None'
        in both cases.

    Returns: A 'Machine' object that describes the current machine, or 'None' if
      no unique match was found and 'raise_exception' is 'False'.
    """
    hostnames = [hostname] if hostname is not None else _local_hostnames()
    tried = []
    matches = []
    for candidate in hostnames:
        tried.append(candidate)
        matches = _machines_matching(candidate, machines_dir)
        if matches:
            break
    if len(matches) == 1:
        return matches[0]
    if not raise_exception:
        return None
    if len(matches) > 1:
        raise UnknownMachineError(
            f"The hostname '{tried[-1]}' matches multiple machines: "
            + ", ".join(machine.Name for machine in matches)
            + ". Make their 'HostnameRegex' patterns more specific."
        )
    raise UnknownMachineError(
        "The hostname "
        + " / ".join(f"'{candidate}'" for candidate in tried)
        + " did not match any known machine. If you are running on a new "
        "machine, please add it to 'support/Machines/' with a 'HostnameRegex' "
        "that matches its hostnames."
    )
