# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def acceleration(x, potential_center, smoothing_parameter, transition_width):
    result = x - potential_center
    r_prime = np.sqrt(np.dot(result, result))
    one_minus_dr_halves = 0.5 * (1.0 - transition_width)
    if (r_prime < 1.e-12):
        for i in range(0, x.size):
            result[i] = -1.0 / (r_prime**2 + smoothing_parameter**2)
    else:
        if (one_minus_dr_halves < r_prime):
            result /= -r_prime**3
        else:
            result /= -(r_prime * (r_prime**2 + smoothing_parameter**2))
    return result


def source_momentum_density(mass_density_cons, momentum_density, x,
                            potential_center, smoothing_parameter,
                            transition_width):
    return (mass_density_cons *
            acceleration(x, potential_center, smoothing_parameter,
                         transition_width))


def source_energy_density(mass_density_cons, momentum_density, x,
                          potential_center, smoothing_parameter,
                          transition_width):
    return np.dot(momentum_density,
                  acceleration(x, potential_center, smoothing_parameter,
                               transition_width))

