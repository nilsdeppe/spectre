# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
from Hydro.SmoothFlow import *


def electron_fraction(
    x,
    t,
    mean_velocity,
    wave_vector,
    pressure,
    adiabatic_index,
    density_amplitude,
):
    return 0.1


def magnetic_field(
    x,
    t,
    mean_velocity,
    wave_vector,
    pressure,
    adiabatic_index,
    density_amplitude,
    magnetic_monopole_amplitude,
):
    return np.array(
        [
            0.0,
            0.0,
            magnetic_monopole_amplitude * x[2] * np.exp(-0.5 * x[2] ** 2),
        ]
    )


def divergence_cleaning_field(
    x,
    t,
    mean_velocity,
    wave_vector,
    pressure,
    adiabatic_index,
    density_amplitude,
    magnetic_monopole_amplitude,
):
    return 0.0
