# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def magnetic_field_dot_spatial_velocity(
    lapse,
    spatial_velocity_one_form,
    lorentz_factor,
    magnetic_field,
    spatial_metric,
):
    return np.dot(magnetic_field, spatial_velocity_one_form)


def lapse_b_over_w(
    lapse,
    spatial_velocity_one_form,
    lorentz_factor,
    magnetic_field,
    spatial_metric,
):
    return lapse * (
        np.einsum("ij, j", spatial_metric, magnetic_field) / lorentz_factor**2
        + spatial_velocity_one_form
        * np.dot(magnetic_field, spatial_velocity_one_form)
    )
