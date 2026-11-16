import csv
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
with open(ROOT_DIR / 'analysis' / 'sawtooth_88653_master_ledger.csv', 'r') as f:
    rows = list(csv.DictReader(f))

# 1. Family summary
fam_stats = defaultdict(lambda: {'SUPPORTED': 0, 'SUGGESTIVE': 0, 'INCONCLUSIVE': 0, 'TOTAL': 0})
for r in rows:
    fam = r['family']
    cl = r['classification']
    fam_stats[fam][cl] = fam_stats[fam].get(cl, 0) + 1
    fam_stats[fam]['TOTAL'] += 1

print('=== FAMILY-LEVEL EVIDENCE BREAKDOWN ===')
print(f'{"Family":12s} {"Supported":>10s} {"Suggestive":>11s} {"Inconclusive":>13s} {"Total":>6s}')
for fam, d in sorted(fam_stats.items(), key=lambda x: -x[1]['SUPPORTED']):
    print(f"{fam:12s} {d['SUPPORTED']:10d} {d['SUGGESTIVE']:11d} {d['INCONCLUSIVE']:13d} {d['TOTAL']:6d}")

# 2. Poloidal Mirnov profile
pmp = [r for r in rows if r['family'] == 'PMP']
print('\n=== POLOIDAL MIRNOV ARRAY (PMP1-14) ===')
for r in sorted(pmp, key=lambda x: int(x['channel_id'].replace('PMP', ''))):
    print(f"{r['channel_id']:6s}: S_deriv={float(r['regime_I_s_deriv']):+6.2f}, SEA_SNR={float(r['regime_I_snr_sea']):4.1f}, Offset={float(r['regime_I_offset_sea_ms']):+5.2f} ms, tau*={float(r['regime_I_tau_star_ms']):+5.2f} ms, class={r['classification']}")

# 3. Divertor probes
div = [r for r in rows if r['family'] == 'DivProArr' and r['classification'] == 'SUPPORTED']
print('\n=== DIVERTOR PROBES (TOP SUPPORTED) ===')
for r in sorted(div, key=lambda x: -float(x['regime_I_snr_sea']))[:12]:
    print(f"{r['channel_id']:12s}: SEA_SNR={float(r['regime_I_snr_sea']):4.1f}, S_deriv={float(r['regime_I_s_deriv']):+6.2f}, Offset={float(r['regime_I_offset_sea_ms']):+5.2f} ms, tau*={float(r['regime_I_tau_star_ms']):+5.2f} ms")

# 4. sx2d Soft X-ray
sx = [r for r in rows if r['family'] == 'sx2d']
print('\n=== SOFT X-RAY TOMOGRAPHY (sx2d) TOP CHORDS ===')
for r in sorted(sx, key=lambda x: -float(x['regime_I_snr_sea']))[:10]:
    print(f"{r['channel_id']:16s}: SEA_SNR={float(r['regime_I_snr_sea']):4.1f}, S_deriv={float(r['regime_I_s_deriv']):+6.2f}, Offset={float(r['regime_I_offset_sea_ms']):+5.2f} ms, tau*={float(r['regime_I_tau_star_ms']):+5.2f} ms, class={r['classification']}")

# 5. Core Density and Heating
print('\n=== CORE & HEATING SIGNALS ===')
for ch_name in ['nave', 'Wp', 'Ip15', 'ECHRG500', 'GASPUFF#2', 'ECE4FAST', 'ECE9FAST', 'ECE10FAST', 'HAFAST7.5']:
    matches = [r for r in rows if r['channel_id'] == ch_name]
    if matches:
        r = matches[0]
        print(f"{r['channel_id']:12s} ({r['family']:10s}): SEA_SNR={float(r['regime_I_snr_sea']):4.1f}, S_deriv={float(r['regime_I_s_deriv']):+6.2f}, Offset={float(r['regime_I_offset_sea_ms']):+5.2f} ms, tau*={float(r['regime_I_tau_star_ms']):+5.2f} ms, class={r['classification']}")

# 6. Pre-mode activity
print('\n=== PRE-MODE ACTIVITY (171.2 - 259.1 ms) ===')
pre_active = [r for r in rows if float(r['pre_mode_event_rate']) > 0.05]
for r in sorted(pre_active, key=lambda x: -float(x['pre_mode_event_rate']))[:10]:
    print(f"{r['channel_id']:16s} ({r['family']:10s}): rate={float(r['pre_mode_event_rate']):.3f} events/ms, pre_skew={float(r['pre_mode_skew']):+5.2f}, class={r['classification']}")

# 7. Regime II Macroscopic correlation
print('\n=== REGIME II MACROSCOPIC ENVELOPE CORRELATION ===')
regime_II = [r for r in rows if abs(float(r['regime_II_main_envelope_r'])) > 0.70 or abs(float(r['regime_II_early_burst_r'])) > 0.70]
for r in sorted(regime_II, key=lambda x: -max(abs(float(x['regime_II_main_envelope_r'])), abs(float(x['regime_II_early_burst_r']))))[:10]:
    print(f"{r['channel_id']:16s} ({r['family']:10s}): Early Burst r={float(r['regime_II_early_burst_r']):+.3f} (tau={float(r['regime_II_early_burst_tau_ms']):+4.1f} ms), Main Env r={float(r['regime_II_main_envelope_r']):+.3f} (tau={float(r['regime_II_main_envelope_tau_ms']):+4.1f} ms)")
