# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def mass_density(x, adiabatic_index, lower_mass_density, upper_mass_density,
                 background_pressure, perturbation_amplitude, damping_factor,
                 interface_height, grav_acceleration):
    return lower_mass_density if x[x.size - 1] < interface_height \
        else upper_mass_density


def velocity(x, adiabatic_index, lower_mass_density, upper_mass_density,
             background_pressure, perturbation_amplitude, damping_factor,
             interface_height, grav_acceleration):
    dim = x.size
    result = np.zeros(dim)
    result[dim - 1] = (perturbation_amplitude *
                       np.sin(2.0 * np.pi * x[0] / 0.5) *
                       np.exp(-((x[dim - 1] - interface_height) /
                                damping_factor)**2))
    return result


def pressure(x, adiabatic_index, lower_mass_density, upper_mass_density,
             background_pressure, perturbation_amplitude, damping_factor,
             interface_height, grav_acceleration):
    lower_specific_weight = lower_mass_density * grav_acceleration
    upper_specific_weight = upper_mass_density * grav_acceleration
    vertical_coord = x[x.size - 1]

    result = background_pressure
    result -= (lower_specific_weight * vertical_coord
               if vertical_coord < interface_height
               else (lower_specific_weight * interface_height +
                     upper_specific_weight *
                     (vertical_coord - interface_height)))
    return result


def specific_internal_energy(x, adiabatic_index, lower_mass_density,
                             upper_mass_density, background_pressure,
                             perturbation_amplitude, damping_factor,
                             interface_height, grav_acceleration):
    return (pressure(x, adiabatic_index, lower_mass_density,
                     upper_mass_density, background_pressure,
                     perturbation_amplitude, damping_factor, interface_height,
                     grav_acceleration) /
            mass_density(x, adiabatic_index, lower_mass_density,
                         upper_mass_density, background_pressure,
                         perturbation_amplitude, damping_factor,
                         interface_height, grav_acceleration) /
            (adiabatic_index - 1.0))


