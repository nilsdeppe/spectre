# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from spectre.support.Logging import configure_logging
from spectre.support.RunSlurmCommand import in_container, run_slurm_command


class TestRunSlurmCommand(unittest.TestCase):
    def setUp(self):
        # '_ssh_identity_file' consults the machine we are running on for its
        # 'ContainerSshKey'. Identifying it resolves the hostname, which can be
        # slow (see 'Machines._fqdn'), and on a machine that defines a key the
        # expected SSH command would depend on where the test runs. Report no
        # machine instead; 'test_ssh_identity_file_from_env' covers the key.
        machine_patcher = patch(
            "spectre.support.RunSlurmCommand.this_machine", return_value=None
        )
        machine_patcher.start()
        self.addCleanup(machine_patcher.stop)

    def test_in_container(self):
        with patch.dict("os.environ", {}, clear=False) as environ:
            environ.pop("SPECTRE_CONTAINER", None)
            self.assertFalse(in_container())
        # The container sets SPECTRE_CONTAINER to "0" (bash-true), so any value
        # (including "0") means we are in a container.
        with patch.dict("os.environ", {"SPECTRE_CONTAINER": "0"}):
            self.assertTrue(in_container())
        with patch.dict("os.environ", {"SPECTRE_CONTAINER": "1"}):
            self.assertTrue(in_container())

    def test_outside_container_runs_locally(self):
        with patch.dict("os.environ", {}, clear=False) as environ:
            environ.pop("SPECTRE_CONTAINER", None)
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=["sacct"], returncode=0, stdout="", stderr=""
                )
                run_slurm_command(
                    ["sacct", "-PX"],
                    cwd=Path("/some/dir"),
                    capture_output=True,
                    text=True,
                )
                mock_run.assert_called_once_with(
                    ["sacct", "-PX"],
                    cwd=Path("/some/dir"),
                    capture_output=True,
                    text=True,
                )

    def test_inside_container_routes_through_ssh(self):
        with patch.dict("os.environ", {"SPECTRE_CONTAINER": "1"}) as environ:
            environ.pop("SPECTRE_CONTAINER_SSH_HOST", None)
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=["ssh"], returncode=0, stdout="", stderr=""
                )
                run_slurm_command(
                    ["sbatch", "Submit.sh"],
                    cwd=Path("/run/dir"),
                    capture_output=True,
                    text=True,
                )
                dispatched_args = mock_run.call_args.args[0]
                self.assertEqual(dispatched_args[0], "ssh")
                self.assertIn("BatchMode=yes", dispatched_args)
                self.assertEqual(dispatched_args[-2], "localhost")
                self.assertEqual(
                    dispatched_args[-1],
                    "cd /run/dir && sbatch Submit.sh",
                )
                # 'cwd' is applied remotely, not on the local ssh process
                self.assertNotIn("cwd", mock_run.call_args.kwargs)

    def test_ssh_host_override(self):
        with patch.dict(
            "os.environ",
            {
                "SPECTRE_CONTAINER": "1",
                "SPECTRE_CONTAINER_SSH_HOST": "head-node",
            },
        ):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=["ssh"], returncode=0, stdout="", stderr=""
                )
                run_slurm_command(["sacct"], capture_output=True, text=True)
                dispatched_args = mock_run.call_args.args[0]
                self.assertEqual(dispatched_args[-2], "head-node")
                # No cwd -> remote command has no leading 'cd'
                self.assertEqual(dispatched_args[-1], "sacct")

    def test_ssh_identity_file_from_env(self):
        with patch.dict(
            "os.environ",
            {
                "SPECTRE_CONTAINER": "1",
                "SPECTRE_CONTAINER_SSH_KEY": "~/.ssh/cluster",
            },
        ):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=["ssh"], returncode=0, stdout="", stderr=""
                )
                run_slurm_command(["sacct"], capture_output=True, text=True)
                dispatched_args = mock_run.call_args.args[0]
                self.assertIn("-i", dispatched_args)
                identity = dispatched_args[dispatched_args.index("-i") + 1]
                # '~' is expanded to the user's home directory
                self.assertEqual(identity, os.path.expanduser("~/.ssh/cluster"))

    def test_missing_ssh_key_message(self):
        with patch.dict("os.environ", {"SPECTRE_CONTAINER": "1"}):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(
                    args=["ssh"],
                    returncode=255,
                    stdout="",
                    stderr="Permission denied (publickey).",
                )
                with self.assertLogs(
                    "spectre.support.RunSlurmCommand", level="ERROR"
                ) as logs:
                    result = run_slurm_command(
                        ["sacct"], capture_output=True, text=True
                    )
                self.assertEqual(result.returncode, 255)
                self.assertTrue(
                    any("$HOME/.ssh" in message for message in logs.output)
                )


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
