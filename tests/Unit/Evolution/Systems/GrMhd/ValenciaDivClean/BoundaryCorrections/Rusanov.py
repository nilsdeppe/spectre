# Distributed under the MIT License.
# See LICENSE.txt for details.

import NonConservativeTerms
import numpy as np


def dg_package_data(
    tilde_b,
    tilde_phi,
    tilde_d,
    tilde_ye,
    tilde_tau,
    tilde_s,
    flux_tilde_b,
    flux_tilde_phi,
    flux_tilde_d,
    flux_tilde_ye,
    flux_tilde_tau,
    flux_tilde_s,
    lapse,
    shift,
    magnetic_field_dot_spatial_velocity,
    lapse_b_over_w,
    spatial_velocity_one_form,
    rest_mass_density,
    electron_fraction,
    temperature,
    spatial_velocity,
    normal_covector,
    normal_vector,
    mesh_velocity,
    normal_dot_mesh_velocity,
    equation_of_state,
):
    def compute_char(lapse_sign):
        return np.abs(
            np.asarray(
                (lapse_sign * lapse - np.dot(shift, normal_covector))
                if normal_dot_mesh_velocity is None
                else (
                    lapse_sign * lapse
                    - np.dot(shift, normal_covector)
                    - normal_dot_mesh_velocity
                )
            )
        )

    return (
        tilde_d,
        tilde_ye,
        tilde_tau,
        tilde_s,
        tilde_b,
        tilde_phi,
        np.asarray(np.dot(flux_tilde_d, normal_covector)),
        np.asarray(np.dot(flux_tilde_ye, normal_covector)),
        np.asarray(np.dot(flux_tilde_tau, normal_covector)),
        np.einsum("ij,i->j", flux_tilde_s, normal_covector),
        np.einsum("ij,i->j", flux_tilde_b, normal_covector),
        np.asarray(np.dot(flux_tilde_phi, normal_covector)),
        np.asarray(np.maximum(compute_char(1.0), compute_char(-1.0))),
    ) + NonConservativeTerms.package_data(
        lapse,
        magnetic_field_dot_spatial_velocity,
        lapse_b_over_w,
        spatial_velocity,
        normal_covector,
    )


def dg_boundary_terms(
    interior_tilde_d,
    interior_tilde_ye,
    interior_tilde_tau,
    interior_tilde_s,
    interior_tilde_b,
    interior_tilde_phi,
    interior_normal_dot_flux_tilde_d,
    interior_normal_dot_flux_tilde_ye,
    interior_normal_dot_flux_tilde_tau,
    interior_normal_dot_flux_tilde_s,
    interior_normal_dot_flux_tilde_b,
    interior_normal_dot_flux_tilde_phi,
    interior_abs_char_speed,
    interior_lapse_times_spatial_velocity,
    interior_normal_dot_lapse_times_spatial_velocity,
    interior_normal_covector,
    interior_lapse_times_magnetic_field_dot_spatial_velocity,
    interior_lapse_b_over_w,
    exterior_tilde_d,
    exterior_tilde_ye,
    exterior_tilde_tau,
    exterior_tilde_s,
    exterior_tilde_b,
    exterior_tilde_phi,
    exterior_normal_dot_flux_tilde_d,
    exterior_normal_dot_flux_tilde_ye,
    exterior_normal_dot_flux_tilde_tau,
    exterior_normal_dot_flux_tilde_s,
    exterior_normal_dot_flux_tilde_b,
    exterior_normal_dot_flux_tilde_phi,
    exterior_abs_char_speed,
    exterior_lapse_times_spatial_velocity,
    exterior_normal_dot_lapse_times_spatial_velocity,
    exterior_normal_covector,
    exterior_lapse_times_magnetic_field_dot_spatial_velocity,
    exterior_lapse_b_over_w,
    use_strong_form,
):
    sign_for_form = 1.0 if use_strong_form else -1.0

    # Use scope and locals() to get arguments into the eval context below
    scope = locals()

    def impl(var_name):
        return np.asarray(
            (
                -0.5
                * (
                    sign_for_form
                    * eval("interior_normal_dot_flux_" + var_name, scope)
                    + eval("exterior_normal_dot_flux_" + var_name, scope)
                )
                - 0.5
                * np.maximum(interior_abs_char_speed, exterior_abs_char_speed)
                * (
                    eval("exterior_" + var_name, scope)
                    - eval("interior_" + var_name, scope)
                )
            )
        )

    (
        tilde_b_term,
        tilde_phi_term,
        tilde_tau_term,
        tilde_s_term,
    ) = NonConservativeTerms.boundary_terms(
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
    )

    return (
        impl("tilde_b") + tilde_b_term,
        np.asarray(impl("tilde_phi") + tilde_phi_term),
        impl("tilde_d"),
        impl("tilde_ye"),
        np.asarray(impl("tilde_tau") + tilde_tau_term),
        impl("tilde_s") + tilde_s_term,
    )
