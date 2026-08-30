#!/usr/bin/env python
"""
    This program visualize Thomson scattering measurement datawhich is saved as a edf-formatted file.

    Usage: tsview [-h] [--version] -f filename [-T id id id id] [-R [id ...]] [-t [id ...]] [-p] [-r]
                            [-M val val]

    options:
        -h, --help            Show this help message and exit
        --version             Show program's version number and exit
        -f filename, --file filename
                              Set an edf-formatted file to be read
        -T id id id id, --Thomson id id id id
                              Set IDs for [Te, dT, ne, dn]
        -R [id ...], --Rslices [id ...]
                              Set IDs of R for R-slices (default 1)
        -t [id ...], --timeslices [id ...]
                              Set IDs of time for time-slices (default 1)
       -p, --portrait         Portrait mode
       -r, --reverse          Reverse color map (default True jet_r)
       -M val val, --Max val val
                              Set [Te_mac, ne_max]

    Example:
        python tsview.py -f 194934/thomson@194934_1.txt -M 1.5 30 -T 1 2 3 4

    Dependence
    ----------
    turnelib.py
    matplotlibutil.py

    Status
    ------
    Version 1.0
    
    Revision History
    ----------------
    [12-Oct-2025] Creation                           ver 0.9
    [12-Oct-2025] Bug in save-part fixed             ver 0.9.1
    [17-Oct-2025] Released                           ver 1.0

    Copyright
    ---------
    2025 Shigeru Inagaki (inagaki.shigeru.7s@kyoto-u.ac.jp)
    Released under the CC BY-NC 4.0.

"""
import sys
import copy
from pathlib import Path
import numpy as np
import turnelib as TE
import matplotlib
import matplotlib.pyplot as plt
import matplotlibutil as MPU
matplotlib.axes.Axes.add_errorbar = MPU.add_errorbar
matplotlib.figure.Figure.make_subplots = MPU.make_subplots

def valnames(y, yname, yunit, ndigit=3):
    names = []
    strformat =  '={:.'+str(ndigit)+'f}'
    if yunit.upper().find('ARB') >= 0:
        for val in y:
           name = yname + strformat.format(val)
           names.append(name)
    else:
        for val in y:
           name = yname + strformat.format(val)+ '(' + yunit + ')'
           names.append(name)
    return names

if __name__ == "__main__":
    version = "1.0"
    timestamp = "2025/10/17"

    myparser = TE.default_parser('tsview', 'Visualize Thomson scattering measurement data', version, timestamp)
    myparser.add_argument(
        '-f',
        '--file',
        action='store',
        required=True,
        type=str,
        metavar = 'filename',
        help='Set an edf-formatted file to be read'
    )
    myparser.add_argument(
        '-T',
        '--Thomson',
        action='store',
        nargs=4,
        type=int,
        default = [1,2,3,4],
        metavar = 'id',
        help='Set IDs for [Te, dT, ne, dn] '
    )
    myparser.add_argument(
        '-R',
        '--Rslices',
        action='store',
        nargs="*",
        type=int,
        default = None,
        metavar = 'id',
        help='Set IDs of R for R-slices (default 1)'
    )
    myparser.add_argument(
        '-t',
        '--timeslices',
        action='store',
        nargs="*",
        type=int,
        default = None,
        metavar = 'id',
        help='Set IDs of time for time-slices (default 1)'
    )
    myparser.add_argument(
        '-p','--portrait', 
        action='store_true',
        default = False,
        help='Portrait mode'
    )
    myparser.add_argument(
        '-r','--reverse', 
        action='store_false',
        default = True,
        help='Reverse color map (default True jet_r)'
    )
    myparser.add_argument(
        '-M','--Max', 
        action='store',
        default = None,
        nargs=2,
        type=float,
        metavar='val',
        help='Set [Te_mac, ne_max]'
    )
    
    args = myparser.parse_args()
    edfdata = TE.edf()    
    dat = edfdata.load(args.file)
    path = Path(args.file)
        
    TdTndn = TE.ensure_array(args.Thomson) + 1
    dim_tevo = 0
    dim_rpro = 1

    if args.Max is None:
        print('Input Te_max (keV),  ne_max (e19 m-3)')
        ans = input('>>>')
        vals = TE.list2float(TE.smart_split(ans))        
        T_max = vals[0]
        n_max = vals[1]
    else:
        T_max = args.Max[0]
        n_max = args.Max[1]
    T_lim = [0,T_max]
    n_lim = 0,n_max
    
    tt = dat[:,dim_tevo].reshape(edfdata.DimSize)
    t = tt[:,0]
    t_name = 't'      #edfdata.DimName[dim_tevo]
    t_unit = edfdata.DimUnit[dim_tevo]
    if t_unit == 'ms' :
        t_unit = 's'
        t = t/1000
    str_t =  valnames(t, t_name, t_unit, ndigit=3)
    RR = dat[:,dim_rpro].reshape(edfdata.DimSize)
    R = RR[0,:]
    R_name = edfdata.DimName[dim_rpro]
    R_unit = edfdata.DimUnit[dim_rpro]
    if R_unit == 'mm' :
        R_unit = 'm'
        R = R/1000
    str_R =  valnames(R, R_name, R_unit, ndigit=3)

    Te =  dat[:,TdTndn[0]].reshape(edfdata.DimSize) 
    dT =  dat[:,TdTndn[1]].reshape(edfdata.DimSize)
    ne =  dat[:,TdTndn[2]].reshape(edfdata.DimSize)
    dummy = dat[:,TdTndn[3]]
    indx = np.where(dummy < 0.0)
    dummy[indx[0]] = 0.0
    dn =  dummy.reshape(edfdata.DimSize)    

    T_unit = edfdata.ValUnit[TdTndn[0]-2]
    n_unit = edfdata.ValUnit[TdTndn[2]-2]

    if T_unit == 'eV' :
        T_unit = 'keV'
        Te = Te/1000
        dT = dT/1000
    if n_unit == '10^16 m^-3':
        n_unit = '10^19 m^-3'
        ne = ne/1000
        dn = dn/1000
        
    if args.Rslices is None :
        id = 1
        for sr in str_R:
            print(id, sr)
            id = id + 1
        print('Select IDs (Max 11)')
        ans = input('>>>')
        vals = TE.list2int(TE.smart_split(ans))
        id_Rs = TE.ensure_array(vals) - 1
    else:
        id_Rs = TE.ensure_array(args.Rslices) - 1
    if len(id_Rs) > 11 :
        print('Please select no more than 11 IDs.')
        sys.exit()

    if args.timeslices is None :
        id = 1
        for st in str_t:
            print(id, st)
            id = id + 1
        print('Select IDs (Max 5)')
        ans = input('>>>')
        vals = TE.list2int(TE.smart_split(ans))
        id_times = TE.ensure_array(vals) - 1
    else:
        id_times = TE.ensure_array(args.timeslices) - 1        
    if len(id_times) > 5 :
        print('Please select no more than 5 IDs.')
        sys.exit()
    
    _left=0.12
    _bottom=0.1
    _right=0.95
    _top=0.95
    _xlen = _right - _left
    _ylen = 0.15
    _offset = 2*_ylen + 0.1
    title = path.name + ' #' + str(edfdata.ShotNo)      
    plots = []
    page = MPU.mypage(fontsize=16, scale=0.85, style='slide', portrait=False)
    nplot_rpro = len(id_times)
    nrows = 2
    ncols = nplot_rpro

    plots = [
        page.add_axes([_left, _top-_ylen, _xlen, _ylen])
    ]
    plots.append(
        page.add_axes([_left, _top-2*_ylen-0.015, _xlen, _ylen],sharex=plots[0])
    )
    plots.extend(
        page.make_subplots(nrows, ncols, shared_xaxes=True, roworder=False, left=_left, bottom=_bottom, right=_right, top=_top-_offset, wspace=0.08,hspace=0.05)
    )

    _colors = MPU.AutoColor(len(id_Rs), reverse=args.reverse)
#    _legend = JC.list_copy_index(str_R, id_Rs)
    _legend = TE. select_elements(str_R, id_Rs)
    time_label = t_name + ' (' + t_unit +')'
    T_label = r'$T_\mathrm{e}\ \mathrm{(keV)}$'
    plots[0].add_errorbar(t, Te[:,id_Rs].transpose(), yerr=dT[:,id_Rs].transpose(), xlabel='', ylabel=T_label, ylim=T_lim, mode='markers', ynames=_legend, colors=_colors,markersize=3)
    plt.setp(plots[0].get_xticklabels(), visible=False) 
    plots[0].legend(_legend,loc='best',frameon=False)
    plots[0].set_title(title, loc='right', fontdict={'fontfamily':'monospace','fontsize':12,})
    
    n_label = r'$n_\mathrm{e}\ \mathrm{(10^{19}\ m^{-3})}$'
    plots[1].add_errorbar(t, ne[:,id_Rs].transpose(), yerr=dn[:,id_Rs].transpose(), xlabel=time_label, ylabel=n_label,  ylim=n_lim, mode='markers', ynames=_legend, colors=_colors,markersize=3)
#    plots[1].legend(_legend,loc='best',frameon=False)

    R_label = r'$R \mathrm{(m)}$' 
    k = 2
    L = k + nplot_rpro
    for i in id_times:
        if k == 2 :
            plots[k].add_errorbar(R, Te[i,:], yerr=dT[i,:], xlabel='', ylabel=T_label, ynames='_nolegend_', ylim=T_lim, mode='markers', colors = 'red',markersize=3)
        else:
            plots[k].add_errorbar(R, Te[i,:], yerr=dT[i,:], xlabel='', ylabel='', ynames='_nolegend_', ylim=T_lim, mode='markers', colors = 'red',markersize=3)
            plt.setp(plots[k].get_yticklabels(), visible=False)
        plt.setp(plots[k].get_xticklabels(), visible=False)
        plots[k].legend(handles=[MPU.legend_label_only(str_t[i])],loc='best',frameon=False)
        if L == 2 + nplot_rpro :
            plots[L].add_errorbar(R, ne[i,:], yerr=dn[i,:], xlabel=R_label, ylabel=n_label, ynames=str_t[i], ylim=n_lim,  mode='markers', colors = 'blue',markersize=3)
        else:
            plots[L].add_errorbar(R, ne[i,:], yerr=dn[i,:], xlabel=R_label, ylabel='', ynames=str_t[i], ylim=n_lim, mode='markers', colors = 'blue',markersize=3)
            plt.setp(plots[L].get_yticklabels(), visible=False)
#        plots[L].legend(str_t[i],loc='best',frameon=False)
        k = k + 1
        L = L + 1
    plt.show()    

    



