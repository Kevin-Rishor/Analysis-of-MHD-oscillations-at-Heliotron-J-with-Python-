
import sys
sys.path.append('c:/TFG/jpack')
sys.path.append('c:/TFG/analysis')
from common.signal_processing import extract_signal_window, compute_cross_spectral_density
from jpack.data_reader import DataReader

dr = DataReader('c:/TFG/data/hj88653')
mp1 = dr.get_channel('MP1')
ece03 = dr.get_channel('ECE03FAST')
ece12 = dr.get_channel('ECE12FAST')
ece13 = dr.get_channel('ECE13FAST')

t_mp1, v_mp1 = extract_signal_window(mp1['time'], mp1['voltage'], 259.1, 275.0)
_, v_ece03 = extract_signal_window(ece03['time'], ece03['voltage'], 259.1, 275.0)
_, v_ece12 = extract_signal_window(ece12['time'], ece12['voltage'], 259.1, 275.0)
_, v_ece13 = extract_signal_window(ece13['time'], ece13['voltage'], 259.1, 275.0)

for ece_name, v_ece in [('ECE03FAST', v_ece03), ('ECE12FAST', v_ece12), ('ECE13FAST', v_ece13)]:
    freqs, _, coh = compute_cross_spectral_density(v_mp1, v_ece, fs=1e6, nperseg=1024, noverlap=512)
    # Band integrated 80-100 kHz
    mask_band = (freqs >= 80000) & (freqs <= 100000)
    band_coh = coh[mask_band].mean()
    # Single-bin 89 kHz
    idx_89 = abs(freqs - 89000).argmin()
    bin_coh = coh[idx_89]
    print(f'{ece_name}: Band (80-100kHz) mean coherence = {band_coh:.4f}, Single bin 89kHz = {bin_coh:.4f}')

