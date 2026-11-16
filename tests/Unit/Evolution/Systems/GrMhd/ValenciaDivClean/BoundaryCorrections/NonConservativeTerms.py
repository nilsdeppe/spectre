# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def package_data(
    lapse,
    magnetic_field_dot_spatial_velocity,
    lapse_b_over_w,
    spatial_velocity,
    normal_covector,
):
    return (
        lapse * spatial_velocity,
        np.asarray(lapse * np.dot(spatial_velocity, normal_covector)),
        normal_covector,
        np.asarray(lapse * magnetic_field_dot_spatial_velocity),
        lapse_b_over_w,
    )


def boundary_terms(
    interior_tilde_b,
    interior_tilde_phi,
    interior_lapse_times_spatial_velocity,
    interior_normal_dot_lapse_times_spatial_velocity,
    interior_normal_covector,
    interior_lapse_times_magnetic_field_dot_spatial_velocity,
    interior_lapse_b_over_w,
    exterior_tilde_b,
    exterior_tilde_phi,
    exterior_lapse_times_spatial_velocity,
    exterior_normal_dot_lapse_times_spatial_velocity,
    exterior_normal_covector,
    exterior_lapse_times_magnetic_field_dot_spatial_velocity,
    exterior_lapse_b_over_w,
):
    """Returns the terms added to the boundary corrections of
    (tilde_b, tilde_phi, tilde_tau, tilde_s)."""
    jump_tilde_phi = exterior_tilde_phi - interior_tilde_phi
    jump_tilde_b = exterior_tilde_b - interior_tilde_b
    interior_tilde_b_dot_normal = np.dot(
        interior_tilde_b, interior_normal_covector
    )
    exterior_tilde_b_dot_normal = np.dot(
        exterior_tilde_b, exterior_normal_covector
    )

    tilde_phi_term = (
        0.25
        * jump_tilde_phi
        * (
            interior_normal_dot_lapse_times_spatial_velocity
            + exterior_normal_dot_lapse_times_spatial_velocity
        )
    )
    tilde_tau_term = 0.25 * jump_tilde_phi * (
        interior_tilde_b_dot_normal + exterior_tilde_b_dot_normal
    ) + 0.25 * (interior_tilde_b_dot_normal - exterior_tilde_b_dot_normal) * (
        interior_lapse_times_magnetic_field_dot_spatial_velocity
        + exterior_lapse_times_magnetic_field_dot_spatial_velocity
    )
    tilde_b_term = 0.25 * (
        np.dot(jump_tilde_b, interior_normal_covector)
        * interior_lapse_times_spatial_velocity
        + np.dot(jump_tilde_b, exterior_normal_covector)
        * exterior_lapse_times_spatial_velocity
    )
    tilde_s_term = 0.25 * (
        np.dot(jump_tilde_b, interior_normal_covector) * interior_lapse_b_over_w
        + np.dot(jump_tilde_b, exterior_normal_covector)
        * exterior_lapse_b_over_w
    )
    return (
        tilde_b_term,
        np.asarray(tilde_phi_term),
        np.asarray(tilde_tau_term),
        tilde_s_term,
    )
