# Distributed under the MIT License.
# See LICENSE.txt for details.

import glob
import logging
import os
import shutil
import sys
import unittest
import xml.etree.ElementTree as ET

import h5py
from click.testing import CliRunner

import spectre.Informer as spectre_informer
from spectre.support.Logging import configure_logging
from spectre.Visualization.GenerateXdmf import (
    generate_xdmf,
    generate_xdmf_command,
)


class TestGenerateXdmf(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(
            spectre_informer.unit_test_src_path(), "Visualization/Python"
        )
        self.test_dir = os.path.join(
            spectre_informer.unit_test_build_path(),
            "Visualization/GenerateXdmf",
        )
        os.makedirs(self.test_dir, exist_ok=True)
        self.maxDiff = None

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_generate_xdmf(self):
        data_files = glob.glob(os.path.join(self.data_dir, "VolTestData*.h5"))
        output_filename = os.path.join(
            self.test_dir, "Test_GenerateXdmf_output"
        )
        generate_xdmf(
            h5files=data_files,
            output=output_filename,
            subfile_name="element_data",
            start_time=0.0,
            stop_time=1.0,
            stride=1,
            coordinates="InertialCoordinates",
        )

        # The script is quite opaque right now, so we only test that we can run
        # it and it produces output without raising an error. To test more
        # details, we should refactor the script into smaller units.
        self.assertTrue(os.path.isfile(output_filename + ".xmf"))

        # Also make sure that the output doesn't change. This has caught many
        # bugs.
        # - Compare canonicalized XML with stripped whitespace. Pretty
        #   indentation was only added in Python 3.9.
        self.assertEqual(
            ET.canonicalize(
                from_file=output_filename + ".xmf", strip_text=True
            ),
            ET.canonicalize(
                from_file=os.path.join(self.data_dir, "VolTestData.xmf"),
                strip_text=True,
            ).replace(
                "VolTestData0.h5",
                os.path.relpath(data_files[0], self.test_dir),
            ),
        )

    def test_tetrahedral_generate_xdmf(self):
        data_files = glob.glob(os.path.join(self.data_dir, "VolTestData*.h5"))
        output_filename = os.path.join(
            self.test_dir, "Test_GenerateXdmf_output"
        )
        generate_xdmf(
            h5files=data_files,
            output=output_filename,
            subfile_name="element_data",
            start_time=0.0,
            stop_time=1.0,
            stride=1,
            coordinates="InertialCoordinates",
            use_tetrahedral_connectivity=True,
        )

        # The script is quite opaque right now, so we only test that we can run
        # it and it produces output without raising an error. To test more
        # details, we should refactor the script into smaller units.
        self.assertTrue(os.path.isfile(output_filename + ".xmf"))

        # Also make sure that the output doesn't change. This has caught many
        # bugs.
        # - Compare canonicalized XML with stripped whitespace. Pretty
        #   indentation was only added in Python 3.9.
        self.assertEqual(
            ET.canonicalize(
                from_file=output_filename + ".xmf", strip_text=True
            ),
            ET.canonicalize(
                from_file=os.path.join(
                    self.data_dir, "VolTestDataTetrahedral.xmf"
                ),
                strip_text=True,
            ).replace(
                "VolTestData0.h5",
                os.path.relpath(data_files[0], self.test_dir),
            ),
        )

    def test_surface_generate_xdmf(self):
        data_files = [os.path.join(self.data_dir, "SurfaceTestData.h5")]
        output_filename = os.path.join(
            self.test_dir, "Test_SurfaceGenerateXdmf_output"
        )
        generate_xdmf(
            h5files=data_files,
            output=output_filename,
            subfile_name="AhA",
            start_time=0.0,
            stop_time=0.03,
            stride=1,
            coordinates="InertialCoordinates",
        )

        # The script is quite opaque right now, so we only test that we can run
        # it and it produces output without raising an error. To test more
        # details, we should refactor the script into smaller units.
        self.assertTrue(os.path.isfile(output_filename + ".xmf"))

    def test_subfile_not_found(self):
        data_files = glob.glob(os.path.join(self.data_dir, "VolTestData*.h5"))
        output_filename = os.path.join(
            self.test_dir, "Test_GenerateXdmf_subfile_not_found"
        )
        with self.assertRaisesRegex(ValueError, "Could not open subfile"):
            generate_xdmf(
                h5files=data_files,
                output=output_filename,
                subfile_name="unknown_subfile",
                start_time=0.0,
                stop_time=1.0,
                stride=1,
                coordinates="InertialCoordinates",
            )

    def test_cli(self):
        data_files = glob.glob(os.path.join(self.data_dir, "VolTestData*.h5"))
        output_filename = os.path.join(
            self.test_dir, "Test_GenerateXdmf_output"
        )
        runner = CliRunner()
        result = runner.invoke(
            generate_xdmf_command,
            [
                *data_files,
                "-o",
                output_filename,
                "-d",
                "element_data",
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)

        # List available subfiles
        result = runner.invoke(
            generate_xdmf_command,
            [
                *data_files,
            ],
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("element_data", result.output)


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
