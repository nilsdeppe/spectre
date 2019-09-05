# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def r_prime(x, disk_center):
    x_prime = x - disk_center
    return np.sqrt(np.dot(x_prime, x_prime))


def mass_density(x, adiabatic_index, ambient_mass_density, ambient_pressure,
                 disk_center, disk_mass_density, disk_inner_radius,
                 disk_outer_radius, smoothing_parameter, transition_width):
    r_p = r_prime(x, disk_center)
    r_inn_minus = disk_inner_radius - 0.5 * transition_width
    r_inn_plus = disk_inner_radius + 0.5 * transition_width
    r_out_minus = disk_outer_radius - 0.5 * transition_width
    r_out_plus = disk_outer_radius + 0.5 * transition_width
    prefactor = (disk_mass_density - ambient_mass_density) / transition_width
    if (r_p > r_inn_minus and r_p <= r_inn_plus):
        return ambient_mass_density + prefactor * (r_p - r_inn_minus)
    elif (r_p > r_inn_plus and r_p <= r_out_minus):
        return disk_mass_density
    elif (r_p > r_out_minus and r_p <= r_out_plus):
        return disk_mass_density - prefactor * (r_p - r_out_minus)
    else:
        return ambient_mass_density


def velocity(x, adiabatic_index, ambient_mass_density, ambient_pressure,
             disk_center, disk_mass_density, disk_inner_radius,
             disk_outer_radius, smoothing_parameter, transition_width):
    result = np.zeros(x.size)
    r_p = r_prime(x, disk_center)
    if (r_p >= disk_inner_radius - 2.0 * transition_width and
        r_p < disk_outer_radius + 2.0 * transition_width):
        prefactor = 1.0 / np.power(r_p, 1.5)
        result[0] = -prefactor * (x[1] - disk_center[1])
        result[1] = prefactor * (x[0] - disk_center[0])
    return result


def pressure(x, adiabatic_index, ambient_mass_density, ambient_pressure,
             disk_center, disk_mass_density, disk_inner_radius,
             disk_outer_radius, smoothing_parameter, transition_width):
    return ambient_pressure


def specific_internal_energy(x, adiabatic_index, ambient_mass_density,
                             ambient_pressure, disk_center, disk_mass_density,
                             disk_inner_radius, disk_outer_radius,
                             smoothing_parameter, transition_width):
    return (pressure(x, adiabatic_index, ambient_mass_density, ambient_pressure,
                     disk_center, disk_mass_density, disk_inner_radius,
                     disk_outer_radius, smoothing_parameter, transition_width) /
            mass_density(x, adiabatic_index, ambient_mass_density,
                         ambient_pressure, disk_center, disk_mass_density,
                         disk_inner_radius, disk_outer_radius,
                         smoothing_parameter, transition_width) /
            (adiabatic_index - 1.0))

