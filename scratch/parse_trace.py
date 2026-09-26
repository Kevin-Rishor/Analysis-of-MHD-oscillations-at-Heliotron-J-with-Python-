with open('c:/TFG/mhd_obj2_88653.log', 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open('c:/TFG/scratch/trace_out_all.txt', 'w', encoding='utf-8') as out:
    for line in lines:
        if '[TRACE] p_adj_m' in line or '[TRACE] Model' in line or 'FREQUENCY BAND' in line:
            out.write(line.strip() + '\n')
