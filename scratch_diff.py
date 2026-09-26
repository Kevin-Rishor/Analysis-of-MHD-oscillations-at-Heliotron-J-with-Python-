import sys
import difflib

def diff_files(f1, f2):
    print(f'\n--- Diff: {f1} vs {f2} ---')
    try:
        lines1 = open(f1, 'r', encoding='utf-8').readlines()
        lines2 = open(f2, 'r', encoding='utf-8').readlines()
        diff = list(difflib.unified_diff(lines1, lines2, fromfile=f1, tofile=f2, n=0))
        if not diff:
            print('Identical.')
        else:
            print(''.join(diff))
    except Exception as e:
        print(f'Error: {e}')

def show_file(f):
    print(f'\n--- File: {f} ---')
    try:
        print(open(f, 'r', encoding='utf-8').read())
    except Exception as e:
        print(f'Error: {e}')

def show_lines(f1, f2, lines):
    print(f'\n--- Line comparison: {f1} vs {f2} ---')
    try:
        lines1 = open(f1, 'r', encoding='utf-8').readlines()
        lines2 = open(f2, 'r', encoding='utf-8').readlines()
        for l in lines:
            idx = l - 1
            l1 = lines1[idx].strip() if idx < len(lines1) else '<EOF>'
            l2 = lines2[idx].strip() if idx < len(lines2) else '<EOF>'
            print(f'Line {l}:')
            print(f'  {f1}: {l1}')
            print(f'  {f2}: {l2}')
    except Exception as e:
        print(f'Error: {e}')

show_file('analysis/obj1_mode_identification/run_obj1.py')
show_file('analysis/obj2_dynamics_and_structure/run_obj2.py')
show_file('analysis/obj3_nonlinear_coupling/run_obj3.py')
show_file('run_all_objectives.py')

diff_files('analysis/mhd_analysis_obj2.py', 'analysis/obj2_dynamics_and_structure/mhd_analysis_obj2.py')
diff_files('analysis/mhd_analysis_obj1.py', 'analysis/obj1_mode_identification/mhd_analysis_obj1.py')
diff_files('analysis/mhd_common.py', 'analysis/common/mhd_common.py')

show_lines('analysis/mhd_analysis_obj2.py', 'analysis/obj2_dynamics_and_structure/mhd_analysis_obj2.py', [117, 120, 278, 498])
