"""
Heliotron J ECE Diagnostic Conversion Module.

Provides exact mapping from ECE microwave receiving frequency (GHz) to normalized minor
radius rho and physical minor radius r (cm), based on official TRAVIS 3D ray-tracing
simulations at corner port #11.5 (Standard Configuration, H+V = +89%).

Fitted in OriginLab using ExpDec1:
    rho(f) = A1 * exp(-f / t1) + y0
    where y0 = -2.24743, A1 = 22.29009, t1 = 30.62389 GHz (R^2 = 0.99971, Red. Chi^2 = 7.02e-5).
"""

import numpy as np

# Official ExpDec1 fit coefficients from Professor (TRAVIS beamtrace_100, Port #11.5, H+V=+89%)
Y0 = -2.24743
A1 = 22.29009
T1 = 30.62389  # GHz
A_MINOR_CM = 17.0  # Average minor radius in cm

def freq_to_rho(f_ghz):
    """
    Convert ECE frequency (GHz) to normalized minor radius rho.
    f_ghz can be a float or a numpy array.
    """
    f_arr = np.asarray(f_ghz, dtype=float)
    return A1 * np.exp(-f_arr / T1) + Y0

def rho_to_freq(rho):
    """
    Convert normalized minor radius rho to ECE frequency (GHz).
    rho can be a float or a numpy array.
    """
    r_arr = np.asarray(rho, dtype=float)
    # Ensure argument to log is positive
    arg = (r_arr - Y0) / A1
    arg = np.clip(arg, 1e-12, None)
    return -T1 * np.log(arg)

def freq_to_r_cm(f_ghz, a_cm=A_MINOR_CM):
    """
    Convert ECE frequency (GHz) to physical minor radius r (cm).
    """
    return freq_to_rho(f_ghz) * a_cm

# 16-channel Heliotron J fast radiometer array specifications
ECE_CHANNELS = {
    1: {'name': 'ECE01FAST', 'f_ghz': 57.5, 'lo_bank': 'A', 'rho': float(freq_to_rho(57.5)), 'tau': 'thin'},
    9: {'name': 'ECE09FAST', 'f_ghz': 58.5, 'lo_bank': 'B', 'rho': float(freq_to_rho(58.5)), 'tau': 'thin'},
    2: {'name': 'ECE02FAST', 'f_ghz': 59.5, 'lo_bank': 'A', 'rho': float(freq_to_rho(59.5)), 'tau': 'thin'},
    10: {'name': 'ECE10FAST', 'f_ghz': 60.5, 'lo_bank': 'B', 'rho': float(freq_to_rho(60.5)), 'tau': 'thin'},
    3: {'name': 'ECE03FAST', 'f_ghz': 61.5, 'lo_bank': 'A', 'rho': float(freq_to_rho(61.5)), 'tau': 'thin'},
    11: {'name': 'ECE11FAST', 'f_ghz': 62.5, 'lo_bank': 'B', 'rho': float(freq_to_rho(62.5)), 'tau': 'thin'},
    4: {'name': 'ECE04FAST', 'f_ghz': 63.5, 'lo_bank': 'A', 'rho': float(freq_to_rho(63.5)), 'tau': 'thick'},
    12: {'name': 'ECE12FAST', 'f_ghz': 64.5, 'lo_bank': 'B', 'rho': float(freq_to_rho(64.5)), 'tau': 'thick'},
    5: {'name': 'ECE05FAST', 'f_ghz': 65.5, 'lo_bank': 'A', 'rho': float(freq_to_rho(65.5)), 'tau': 'thick'},
    13: {'name': 'ECE13FAST', 'f_ghz': 66.5, 'lo_bank': 'B', 'rho': float(freq_to_rho(66.5)), 'tau': 'thick'},
    6: {'name': 'ECE06FAST', 'f_ghz': 67.5, 'lo_bank': 'A', 'rho': float(freq_to_rho(67.5)), 'tau': 'thick'},
    14: {'name': 'ECE14FAST', 'f_ghz': 68.5, 'lo_bank': 'B', 'rho': float(freq_to_rho(68.5)), 'tau': 'thick'},
    7: {'name': 'ECE07FAST', 'f_ghz': 69.5, 'lo_bank': 'A', 'rho': float(freq_to_rho(69.5)), 'tau': 'thick'},
    15: {'name': 'ECE15FAST', 'f_ghz': 70.5, 'lo_bank': 'B', 'rho': float(freq_to_rho(70.5)), 'tau': 'thick'},
    8: {'name': 'ECE08FAST', 'f_ghz': 71.5, 'lo_bank': 'A', 'rho': float(freq_to_rho(71.5)), 'tau': 'thick'},
    16: {'name': 'ECE16FAST', 'f_ghz': 72.5, 'lo_bank': 'B', 'rho': float(freq_to_rho(72.5)), 'tau': 'thick'},
}

if __name__ == '__main__':
    print(f"Heliotron J ECE Diagnostics (TRAVIS ExpDec1 Fit)")
    print(f"{'Channel':<10} | {'f (GHz)':<8} | {'rho':<8} | {'r (cm)':<8} | {'Optical Depth':<15}")
    print("-" * 55)
    for ch_id in sorted(ECE_CHANNELS.keys(), key=lambda k: ECE_CHANNELS[k]['f_ghz']):
        info = ECE_CHANNELS[ch_id]
        r_cm = info['rho'] * A_MINOR_CM
        tau_str = 'tau >= 1 (thick)' if info['tau'] == 'thick' else 'tau < 1 (thin)'
        print(f"{info['name']:<10} | {info['f_ghz']:<8.1f} | {info['rho']:<8.4f} | {r_cm:<8.2f} | {tau_str:<15}")
