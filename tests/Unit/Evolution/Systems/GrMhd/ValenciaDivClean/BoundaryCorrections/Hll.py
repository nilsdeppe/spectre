# Distributed under the MIT License.
# See LICENSE.txt for details.

import NonConservativeTerms
import numpy as np

_atmosphere_density_cutoff = 1.0e-8
_magnetic_field_magnitude_for_hydro = 1.0e-30
_polytropic_constant = 100.0
_polytropic_gamma = 2.0


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
        magnetic_field_magnitude = np.sqrt(np.einsum("i,i->", tilde_b, tilde_b))
        if (
            magnetic_field_magnitude < _magnetic_field_magnitude_for_hydro
            and rest_mass_density > _atmosphere_density_cutoff
        ):
            # Sound speeds
            pressure = (
                _polytropic_constant * rest_mass_density**_polytropic_gamma
            )
            sound_speed_squared = min(
                max(
                    (
                        _polytropic_gamma
                        * (_polytropic_gamma - 1.0)
                        * pressure
                        / (
                            rest_mass_density * (_polytropic_gamma - 1.0)
                            + _polytropic_gamma * pressure
                        )
                    ),
                    0.0,
                ),
                1.0,
            )
            # As in the C++: v^2 is clamped, v.n is not.
            velocity_dot_normal = np.einsum(
                "i,i->", spatial_velocity, normal_covector
            )
            velocity_squared = min(
                max(
                    np.einsum(
                        "i,i->", spatial_velocity, spatial_velocity_one_form
                    ),
                    0.0,
                ),
                1.0 - 1.0e-8,
            )
            one_over_lorentz_factor_squared = 1.0 - velocity_squared
            d = np.sqrt(
                max(
                    0.0,
                    sound_speed_squared
                    * one_over_lorentz_factor_squared
                    * (
                        1.0
                        - velocity_squared * sound_speed_squared
                        - velocity_dot_normal**2 * (1.0 - sound_speed_squared)
                    ),
                )
            )
            Lambda = (
                lapse
                / (1.0 - velocity_squared * sound_speed_squared)
                * (
                    velocity_dot_normal * (1.0 - sound_speed_squared)
                    + lapse_sign * d
                )
            )
            return np.asarray(
                (Lambda - np.dot(shift, normal_covector))
                if normal_dot_mesh_velocity is None
                else (
                    Lambda
                    - np.dot(shift, normal_covector)
                    - normal_dot_mesh_velocity
                )
            )
        else:
            # Light speeds
            return np.asarray(
                (lapse_sign * lapse - np.dot(shift, normal_covector))
                if normal_dot_mesh_velocity is None
                else (
                    lapse_sign * lapse
                    - np.dot(shift, normal_covector)
                    - normal_dot_mesh_velocity
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
        compute_char(1.0),
        compute_char(-1.0),
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
    interior_largest_outgoing_char_speed,
    interior_largest_ingoing_char_speed,
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
    exterior_largest_outgoing_char_speed,
    exterior_largest_ingoing_char_speed,
    exterior_lapse_times_spatial_velocity,
    exterior_normal_dot_lapse_times_spatial_velocity,
    exterior_normal_covector,
    exterior_lapse_times_magnetic_field_dot_spatial_velocity,
    exterior_lapse_b_over_w,
    use_strong_form,
):
    lambda_max = np.maximum(
        0.0,
        np.maximum(
            interior_largest_outgoing_char_speed,
            -exterior_largest_ingoing_char_speed,
        ),
    )
    lambda_min = np.minimum(
        0.0,
        np.minimum(
            interior_largest_ingoing_char_speed,
            -exterior_largest_outgoing_char_speed,
        ),
    )

    lambda_interior = lambda_min if use_strong_form else lambda_max

    # Use scope and locals() to get arguments into the eval context below
    scope = locals()

    def impl(var_name):
        return np.asarray(
            (
                lambda_interior
                * eval("interior_normal_dot_flux_" + var_name, scope)
                + lambda_min
                * eval("exterior_normal_dot_flux_" + var_name, scope)
                + lambda_max
                * lambda_min
                * (
                    eval("exterior_" + var_name, scope)
                    - eval("interior_" + var_name, scope)
                )
            )
            / (lambda_max - lambda_min)
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
