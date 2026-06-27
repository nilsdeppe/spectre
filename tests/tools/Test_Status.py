# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os
import shutil
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import yaml

import spectre.IO.H5 as spectre_h5
import spectre.tools.Status.Status as status_module
from spectre.Informer import unit_test_build_path
from spectre.support.Logging import configure_logging
from spectre.tools.Status.ExecutableStatus import match_executable_status
from spectre.tools.Status.Status import (
    fetch_job_data,
    get_executable_name,
    get_input_file,
    supported_sacct_fields,
)


class TestExecutableStatus(unittest.TestCase):
    def setUp(self):
        self.input_file = {
            "Observers": {"ReductionFileName": "Reductions"},
            "EventsAndTriggersAtSlabs": [
                {
                    "Trigger": "Always",
                    "Events": [
                        {"ObserveTimeStep": {"SubfileName": "TimeSteps"}}
                    ],
                }
            ],
        }
        self.work_dir = os.path.join(
            unit_test_build_path(), "tools/ExecutableStatus"
        )
        shutil.rmtree(self.work_dir, ignore_errors=True)
        os.makedirs(self.work_dir, exist_ok=True)
        with spectre_h5.H5File(
            os.path.join(self.work_dir, "Reductions.h5"), "w"
        ) as open_h5_file:
            # Time data
            time_subfile = open_h5_file.insert_dat(
                "/TimeSteps",
                legend=[
                    "Time",
                    "Slab size",
                    "Minimum Walltime",
                    "Maximum Walltime",
                ],
                version=0,
            )
            time_subfile.append([0.0, 0.5, 0.5, 1.0])
            time_subfile.append([1.0, 0.5, 2.0, 3.0])
            time_subfile.append([2.0, 0.5, 3.0, 4.0])
            open_h5_file.close_current_object()
            # Control systems data
            rotation_subfile = open_h5_file.insert_dat(
                "/ControlSystems/Rotation/z",
                legend=["FunctionOfTime"],
                version=0,
            )
            rotation_subfile.append([0.0])
            rotation_subfile.append([np.pi])
            open_h5_file.close_current_object()
            # AH data
            aha_subfile = open_h5_file.insert_dat(
                "/ApparentHorizons/ControlSystemAhA_Centers",
                legend=[
                    "InertialCenter_x",
                    "InertialCenter_y",
                    "InertialCenter_z",
                ],
                version=0,
            )
            aha_subfile.append([1.0, 0.0, 0.0])
            open_h5_file.close_current_object()
            ahb_subfile = open_h5_file.insert_dat(
                "/ApparentHorizons/ControlSystemAhB_Centers",
                legend=[
                    "InertialCenter_x",
                    "InertialCenter_y",
                    "InertialCenter_z",
                ],
                version=0,
            )
            ahb_subfile.append([-1.0, 0.0, 0.0])
            open_h5_file.close_current_object()
            # Constraints
            constraints_subfile = open_h5_file.insert_dat(
                "/Norms",
                legend=[
                    "L2Norm(ConstraintEnergy)",
                    "L2Norm(PointwiseL2Norm(ThreeIndexConstraint))",
                ],
                version=0,
            )
            constraints_subfile.append([1.0e-3, 1.0e-4])
            open_h5_file.close_current_object()
            # Elliptic solver residuals
            residuals_subfile = open_h5_file.insert_dat(
                "/NewtonRaphsonResiduals",
                legend=["Iteration", "Residual"],
                version=0,
            )
            residuals_subfile.append([0.0, 1.0e-1])
            residuals_subfile.append([1.0, 1.0e-4])
            open_h5_file.close_current_object()
            residuals_subfile = open_h5_file.insert_dat(
                "/GmresResiduals",
                legend=["Iteration", "Residual"],
                version=0,
            )
            residuals_subfile.append([0.0, 1.0e-1])
            residuals_subfile.append([1.0, 1.0e-4])
            residuals_subfile.append([0.0, 1.0e-2])
            residuals_subfile.append([1.0, 1.0e-5])

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_evolution_status(self):
        executable_status = match_executable_status("EvolveSomething")
        status = executable_status.status(self.input_file, self.work_dir)
        self.assertEqual(status, {"Time": 2.0, "Speed": 2400.0})
        self.assertEqual(executable_status.format("Time", 1.5), "1.5")
        self.assertEqual(executable_status.format("Speed", 1.2), "1.2")

    def test_evolve_bbh_status(self):
        executable_status = match_executable_status("EvolveGhBinaryBlackHole")
        status = executable_status.status(self.input_file, self.work_dir)
        self.assertEqual(status["Time"], 2.0)
        self.assertEqual(status["Speed"], 2400.0)
        self.assertEqual(status["Orbits"], 0.5)
        self.assertEqual(status["Separation"], 2.0)
        self.assertEqual(status["3-Index Constraint"], 1.0e-4)

    def test_evolve_single_bh_status(self):
        executable_status = match_executable_status("EvolveGhSingleBlackHole")
        status = executable_status.status(self.input_file, self.work_dir)
        self.assertEqual(status["Time"], 2.0)
        self.assertEqual(status["Speed"], 2400.0)
        self.assertEqual(status["Constraint Energy"], 1.0e-3)

    def test_elliptic_status(self):
        executable_status = match_executable_status("SolveSomething")
        status = executable_status.status(self.input_file, self.work_dir)
        self.assertEqual(status, {"Iteration": 1.0, "Residual": 1.0e-4})
        self.assertEqual(executable_status.format("Iteration", 1.0), "1")
        self.assertEqual(
            executable_status.format("Residual", 1.0e-4), "1.0e-04"
        )

    def test_xcts_status(self):
        executable_status = match_executable_status("SolveXcts")
        status = executable_status.status(self.input_file, self.work_dir)
        self.assertEqual(
            status,
            {
                "Nonlinear iteration": 1,
                "Nonlinear residual": 1.0e-4,
                "Linear iteration": 2,
                "Linear residual": 1.0e-5,
            },
        )
        self.assertEqual(
            executable_status.format("Nonlinear iteration", 1.0), "1"
        )
        self.assertEqual(executable_status.format("Linear iteration", 1.0), "1")
        self.assertEqual(
            executable_status.format("Nonlinear residual", 1.0e-4), "1.0e-04"
        )
        self.assertEqual(
            executable_status.format("Linear residual", 1.0e-4), "1.0e-04"
        )


class TestStatus(unittest.TestCase):
    def setUp(self):
        self.slurm_comment = (
            "SPECTRE_INPUT_FILE=path/to/input/file\n"
            "SPECTRE_EXECUTABLE=path/to/executable\n"
        )
        self.work_dir = os.path.join(unit_test_build_path(), "tools/Status")
        self.input_file_path = os.path.join(self.work_dir, "InputFile.yaml")
        os.makedirs(self.work_dir, exist_ok=True)
        with open(self.input_file_path, "w") as open_input_file:
            yaml.safe_dump_all([{"Executable": "MyExec"}, {}], open_input_file)

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_get_input_file(self):
        self.assertEqual(
            get_input_file(self.slurm_comment, self.work_dir),
            os.path.join(self.work_dir, "path/to/input/file"),
        )
        self.assertIsNone(
            get_input_file("", os.path.join(self.work_dir, "nonexistent"))
        )
        self.assertEqual(
            get_input_file("", self.work_dir), self.input_file_path
        )

    def test_get_executable_name(self):
        self.assertEqual(
            get_executable_name(self.slurm_comment, self.input_file_path),
            "executable",
        )
        self.assertIsNone(get_executable_name("", None))
        self.assertEqual(
            get_executable_name("", self.input_file_path), "MyExec"
        )

    def test_supported_sacct_fields(self):
        supported_sacct_fields.cache_clear()
        self.addCleanup(supported_sacct_fields.cache_clear)
        with patch.object(
            status_module, "run_slurm_command"
        ) as mock_run_slurm_command:
            mock_run_slurm_command.return_value = subprocess.CompletedProcess(
                args=["sacct", "--helpformat"],
                returncode=0,
                stdout="JobID  State  End\nWorkDir  Comment\n",
                stderr="",
            )
            fields = supported_sacct_fields()
        self.assertEqual(
            fields, frozenset({"JobID", "State", "End", "WorkDir", "Comment"})
        )
        # 'StdOut'/'StdErr' are unavailable on this (mocked) machine
        self.assertNotIn("StdErr", fields)

    def test_supported_sacct_fields_query_fails(self):
        supported_sacct_fields.cache_clear()
        self.addCleanup(supported_sacct_fields.cache_clear)
        with patch.object(
            status_module, "run_slurm_command"
        ) as mock_run_slurm_command:
            mock_run_slurm_command.return_value = subprocess.CompletedProcess(
                args=["sacct", "--helpformat"],
                returncode=1,
                stdout="",
                stderr="sacct: command not found",
            )
            # A failed query returns None so callers don't filter
            self.assertIsNone(supported_sacct_fields())

    def test_fetch_job_data_skips_unsupported_fields(self):
        supported_sacct_fields.cache_clear()
        self.addCleanup(supported_sacct_fields.cache_clear)
        with patch.object(
            status_module,
            "supported_sacct_fields",
            return_value=frozenset({"JobID", "State", "WorkDir"}),
        ), patch.object(
            status_module, "run_slurm_command"
        ) as mock_run_slurm_command:
            mock_run_slurm_command.return_value = subprocess.CompletedProcess(
                args=["sacct"],
                returncode=0,
                stdout="JobID|State|WorkDir\n123|COMPLETED|/run/dir\n",
                stderr="",
            )
            job_data = fetch_job_data(
                ["JobID", "State", "WorkDir", "StdOut", "StdErr"]
            )
        # Only supported fields are passed to 'sacct --format'
        dispatched = mock_run_slurm_command.call_args.args[0]
        format_arg = dispatched[dispatched.index("--format") + 1]
        self.assertEqual(format_arg, "JobID,State,WorkDir")
        # Unsupported fields are re-added as empty columns
        self.assertIn("StdOut", job_data.columns)
        self.assertIn("StdErr", job_data.columns)
        self.assertTrue(job_data["StdErr"].isnull().all())
        self.assertEqual(job_data["State"].iloc[0], "COMPLETED")


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
