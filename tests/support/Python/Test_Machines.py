# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import tempfile
import threading
import unittest
from unittest.mock import patch

import yaml

from spectre.Informer import unit_test_build_path
from spectre.support.Machines import (
    Machine,
    UnknownMachineError,
    _fqdn,
    all_machines,
    this_machine,
    this_machine_by_hostname,
)


def _write_machine(directory, name, hostname_regex=None):
    machine = dict(
        Name=name,
        Description="Just for testing",
        DefaultTasksPerNode=2,
        DefaultProcsPerTask=15,
        DefaultQueue="production",
        DefaultTimeLimit="1-00:00:00",
        LaunchCommandSingleNode=["mpirun", "-n", "2"],
        LaunchCommandLoginNode=["mpirun", "-n", "1"],
    )
    if hostname_regex is not None:
        machine["HostnameRegex"] = hostname_regex
    machinefile_path = os.path.join(directory, name + ".yaml")
    with open(machinefile_path, "w") as open_machinefile:
        yaml.safe_dump(dict(Machine=machine), open_machinefile)
    return machinefile_path


class TestMachines(unittest.TestCase):
    def setUp(self):
        self.machinefile_path = _write_machine(
            unit_test_build_path(), "TestMachine"
        )

        # Don't consult the host these tests happen to run on: its real hostname
        # could match a real machine, and resolving the fully-qualified name can
        # be very slow (see 'Machines._fqdn'). Stand in a name that matches
        # nothing, and skip the lookup entirely. Tests that exercise the
        # hostname-matching itself patch these again with what they need.
        for target, value in [
            ("socket.gethostname", "testhost"),
            ("spectre.support.Machines._fqdn", None),
        ]:
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_this_machine(self):
        # Use an empty machines_dir so the hostname-based fallback can't match
        # the real machine the test is running on.
        with tempfile.TemporaryDirectory() as empty_dir:
            with self.assertRaises(UnknownMachineError):
                this_machine(
                    "NonexistentMachinefile.yaml", machines_dir=empty_dir
                )
            self.assertIsNone(
                this_machine(
                    "NonexistentMachinefile.yaml",
                    raise_exception=False,
                    machines_dir=empty_dir,
                )
            )
        machine = this_machine(self.machinefile_path)
        self.assertIsInstance(machine, Machine)
        self.assertEqual(machine.Name, "TestMachine")
        self.assertEqual(machine.Description, "Just for testing")
        self.assertEqual(machine.DefaultTasksPerNode, 2)
        self.assertEqual(machine.DefaultProcsPerTask, 15)
        self.assertEqual(machine.DefaultQueue, "production")
        self.assertEqual(machine.DefaultTimeLimit, "1-00:00:00")
        self.assertEqual(machine.LaunchCommandSingleNode, ["mpirun", "-n", "2"])
        self.assertEqual(machine.LaunchCommandLoginNode, ["mpirun", "-n", "1"])
        self.assertIsNone(machine.HostnameRegex)
        self.assertIsNone(machine.LoginNodeRegex)
        self.assertEqual(
            machine.launch_command,
            (
                ["mpirun", "-n", "2"]
                if machine.on_compute_node()
                else ["mpirun", "-n", "1"]
            ),
        )

    def test_this_machine_by_hostname(self):
        with tempfile.TemporaryDirectory() as machines_dir:
            _write_machine(machines_dir, "Alpha", hostname_regex=r"^alpha\d+$")
            _write_machine(machines_dir, "Beta", hostname_regex=r"^beta\d+$")
            # Has no HostnameRegex, so it should never match.
            _write_machine(machines_dir, "Gamma")

            # 'HostnameRegex' round-trips onto the 'Machine' object.
            machines_by_name = {
                machine.Name: machine for machine in all_machines(machines_dir)
            }
            self.assertEqual(
                machines_by_name["Alpha"].HostnameRegex, r"^alpha\d+$"
            )
            self.assertIsNone(machines_by_name["Gamma"].HostnameRegex)

            # Unique match.
            match = this_machine_by_hostname(
                "beta42", machines_dir=machines_dir
            )
            self.assertEqual(match.Name, "Beta")

            # No match.
            with self.assertRaises(UnknownMachineError):
                this_machine_by_hostname(
                    "unknownhost", machines_dir=machines_dir
                )
            self.assertIsNone(
                this_machine_by_hostname(
                    "unknownhost",
                    machines_dir=machines_dir,
                    raise_exception=False,
                )
            )

            # Ambiguous match raises, but returns None when exceptions are
            # suppressed.
            _write_machine(machines_dir, "Beta2", hostname_regex=r"^beta\d+$")
            with self.assertRaises(UnknownMachineError):
                this_machine_by_hostname("beta42", machines_dir=machines_dir)
            self.assertIsNone(
                this_machine_by_hostname(
                    "beta42",
                    machines_dir=machines_dir,
                    raise_exception=False,
                )
            )

    def test_this_machine_by_hostname_candidates(self):
        # The plain hostname is matched first and the fully-qualified name is
        # only resolved if it identifies nothing, because that lookup can be
        # slow (see 'test_fqdn_lookup_timeout').
        with tempfile.TemporaryDirectory() as machines_dir:
            _write_machine(machines_dir, "Short", hostname_regex=r"^shorthost$")
            _write_machine(
                machines_dir, "Qualified", hostname_regex=r"^host\.domain$"
            )

            # Plain hostname matches, so we never pay for the lookup.
            with patch("socket.gethostname", return_value="shorthost"):
                with patch("spectre.support.Machines._fqdn") as mock_fqdn:
                    machine = this_machine_by_hostname(
                        machines_dir=machines_dir
                    )
                    self.assertEqual(machine.Name, "Short")
                    mock_fqdn.assert_not_called()

            with patch("socket.gethostname", return_value="host"):
                # Plain hostname matches nothing, so fall back to the
                # fully-qualified name.
                with patch(
                    "spectre.support.Machines._fqdn",
                    return_value="host.domain",
                ):
                    machine = this_machine_by_hostname(
                        machines_dir=machines_dir
                    )
                    self.assertEqual(machine.Name, "Qualified")
                # A lookup that timed out yields 'None' and is skipped. Both
                # hostnames tried are named in the error message.
                with patch("spectre.support.Machines._fqdn", return_value=None):
                    self.assertIsNone(
                        this_machine_by_hostname(
                            machines_dir=machines_dir, raise_exception=False
                        )
                    )
                with patch(
                    "spectre.support.Machines._fqdn", return_value="host.other"
                ):
                    with self.assertRaisesRegex(
                        UnknownMachineError, "'host' / 'host.other'"
                    ):
                        this_machine_by_hostname(machines_dir=machines_dir)

    def test_fqdn_lookup_timeout(self):
        # A reverse-name lookup that never returns (the situation on hosts whose
        # hostname has no DNS record) must not block us: '_fqdn' gives up after
        # its timeout and reports that it found nothing.
        blocked = threading.Event()
        # Release the lookup thread even if the assertion below fails.
        self.addCleanup(blocked.set)

        def hanging_getfqdn():
            blocked.wait()
            return "never.returned"

        _fqdn.cache_clear()
        self.addCleanup(_fqdn.cache_clear)
        with patch("socket.getfqdn", side_effect=hanging_getfqdn):
            self.assertIsNone(_fqdn(timeout=0.01))

    def test_this_machine_ambiguous_hostname(self):
        # When no machine was selected at build time and the hostname matches
        # multiple machines, 'this_machine' should surface the specific
        # ambiguous-configuration error rather than a generic "no match".
        with tempfile.TemporaryDirectory() as machines_dir:
            _write_machine(machines_dir, "AnyA", hostname_regex=r".*")
            _write_machine(machines_dir, "AnyB", hostname_regex=r".*")
            with self.assertRaisesRegex(
                UnknownMachineError, "matches multiple machines"
            ):
                this_machine(
                    "NonexistentMachinefile.yaml", machines_dir=machines_dir
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
