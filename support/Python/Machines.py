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

import os
from dataclasses import dataclass
from typing import List

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

    These optional attributes are read by CMake and the `configure` and
    `compile` scripts in the repository root, which only support POSIX
    extended regular expressions (no Python-only syntax such as `\\d`):

      HostnameRegex: Regex matching the hostnames of all nodes of the machine.
        Used to identify the machine when configuring with CMake.
      LoginNodeRegex: Regex matching the hostnames of the login nodes. Used
        by `./configure` to identify the machine.
      BuildQueue: If set, `./compile` builds on a compute node in this queue
        instead of on the login node.
      BuildCpus: Number of cores `./compile` requests for building. Required
        if `BuildQueue` is set.
      BuildTimeLimit: Wall time limit for `./compile` builds. Required if
        `BuildQueue` is set.
      ContainerImage: Apptainer image that `./compile` builds in by default,
        e.g. "sxscollaboration/spectre:dev". It is pulled to `spectre.sif` in
        the SpECTRE repository.
        Images without a scheme such as `docker://` are pulled from Docker
        Hub.
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
) -> Machine:
    """Determine the machine we are running on.

    Specify the 'MACHINE' option in the CMake build configuration to select a
    machine, or pass the 'machinefile_path' argument to this function.

    Arguments:
      machinefile_path: Path to a YAML file that describes the current machine.
        Defaults to the machine selected in the CMake build configuration.
      raise_exception: If 'True' (default), raises an 'UnknownMachineError' if
        no machine was selected. Otherwise, returns 'None' if no machine was
        selected.

    Returns: A 'Machine' object that describes the current machine, or 'None' if
      no machine was selected and 'raise_exception' is 'False'.
    """
    if not os.path.exists(machinefile_path):
        if not raise_exception:
            return None
        raise UnknownMachineError(
            "No machine was selected. Specify the 'MACHINE' option when "
            "configuring the build with CMake. If you are running on a new "
            "machine, please add it to 'support/Machines/'. The machine file "
            f"was expected at the following path:\n  {machinefile_path}"
        )
    with open(machinefile_path, "r") as open_machinefile:
        return yaml.safe_load(open_machinefile)["Machine"]
