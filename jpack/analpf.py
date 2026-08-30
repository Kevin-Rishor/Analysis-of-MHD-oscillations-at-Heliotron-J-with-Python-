#!/usr/bin/env python
"""
    This application applys the low-passed-filter to a time-series-data.
    The time-series-data should be given as "edf" format.

    Usage:
    analpf [-h] [--version] [-f filename] [-t id] [-c fc] [-o nord] [-w]


    optional arguments:
        -h, --help             show this help message and exit
        --version              show program's version number and exit
        -f filename, --file filename
                               Set an edf-formatted file to be read
        -t id, --target id     Set variables IDs to be plotted (default 1)
        -c fc, --cut fc        Set cut-off frequency for low-pass-filter in [kHz]
        -o nord, --order nord
                               Set order for Bessel filter (default=4)
        -w, --write            Save results to files

    Example:
        python analpf.py -f ../expdata/FY2022/MP3@83489.edf -c 1

    Dependence
    ----------
    turnelib.py
    libana_signal.py
    matplotlibutil.py

    Status
    ------
    Version 1.0

    Authour
    -------
    Shigeru Inagaki                                       
    Institute of Advanced Energy, Kyoto University
    inagaki.shigeru.7s@kyoto-u.ac.jp
    
    Revision History
    ----------------
    [29-Nov-2022] Creation                                  ver 0.9.3
    [11-Oct-2023] Revised                                   ver 0.9.4
    [06-Oct-2023] Bug in save file fixed                    ver 0.9.5
    [12-Nov-2025] -n option removed, -w option added        ver 1.0

    Copyright
    ---------
    2023 Shigeru Inagaki (inagaki.shigeru.7s@kyoto-u.ac.jp)
    Released under the CC BY-NC 4.0.

"""

import os
import math
from pathlib import Path
import scipy.signal as dsp
import libana_signal as LAS
import numpy as np
import turnelib as TE
import matplotlib
import matplotlib.pyplot as plt
import matplotlibutil as MPU
from matplotlib.colors import LogNorm
from matplotlib.widgets import TextBox, Button

matplotlib.axes.Axes.add_trace = MPU.add_trace
matplotlib.axes.Axes.add_errorbar = MPU.add_errorbar
matplotlib.axes.Axes.matview = MPU.matview
matplotlib.figure.Figure.make_subplots = MPU.make_subplots

def LowpassFilter(xin, fsamp, fc, nord=4):
    twopi = 2*np.pi
    omega_c = twopi*fc
    A_numerator, A_denominator =  dsp.bessel(nord, omega_c, 'lowpass', analog=True)
    D_numerator, D_denominator = dsp.bilinear(A_numerator, A_denominator, fsamp)
    filtered = dsp.filtfilt(D_numerator, D_denominator, xin)
    return filtered
    
def submit(text):
    ydata = eval(text)
    l.set_ydata(ydata)
    ax.set_ylim(np.min(ydata), np.max(ydata))
    plt.draw()

my_round_int = lambda x: int((x * 2 + 1) // 2)

if __name__ == "__main__":
    version = "1.0"
    timestamp = "2025/11/12"

    myparser = TE.default_parser('analpf', 'Applying the low-pass-filter', version, timestamp)
    myparser.add_argument(
        '-f',
        '--file',
        action='store',
        type=str,
        default = None,
        metavar = 'filename',
        help='Set an edf-formatted file to be read'
    )
    myparser.add_argument(
        '-t',
        '--target',
        action='store',
        type=int,
        default = 1,
        metavar = 'id',
        help='Set variables IDs to be plotted (default 1)'
    )
    myparser.add_argument(
        '-c',
        '--cut',
        action='store',
        type=float,
        default = None,
        metavar = 'fc',
        help='Set cut-off frequency for low-pass-filter in [kHz]'
    )
    myparser.add_argument(
        '-o',
        '--order',
        action='store',
        type=int,
        default = 4,
        metavar = 'nord',
        help='Set order for Bessel filter (default=4)'
    )
    myparser.add_argument(
        '-w',
        '--write',
        action='store_true',
        default = False,
        help='Save results to files'
    )  
    args = myparser.parse_args()


#--- Initial Setting ---
    edfdata = TE.edf()    
    if args.file is None :
        dat = edfdata.load()
        title = '#' + edfdata.Name + '@' + str(edfdata.ShotNo)
    else:
        dat = edfdata.load(args.file)
        p = Path(args.file)
        title = p.stem
    target = args.target 

    t = dat[:,0]
    ys = dat[:,target]
    if edfdata.ValNo == 1 :
        _ylabel = edfdata.Name + '({})'.format(edfdata.ValUnit[target-1])
    else :
        _ylabel = edfdata.ValName[target-1] + '({})'.format(edfdata.ValUnit[target-1])

    if edfdata.DimUnit[0] == 'ms' :
        t = t/1000.0  #ms-->sec
    dt = (t[100]-t[0])/100

    kHz = 1.0e3
    filtered = LowpassFilter(ys, 1/dt, args.cut*kHz, args.order)
    
    page = MPU.mypage()
    plots = page.make_subplots(1,1)
    plots[0].add_trace(t*1000, ys, xlabel='t (ms)', ylabel=_ylabel, mode='lines', ynames='Raw', colors='black')
    plots[0].add_trace(t*1000, filtered, xlabel='t (ms)', ylabel=_ylabel, mode='lines', ynames='Filtered', colors='red')    
    plots[0].set_title(title, loc='right', fontdict={'fontfamily':'monospace','fontsize':12,})
    plt.show()

    if args.write :
        if edfdata.DimUnit[0] == 'ms' :
            t = t*1000.0  #sec-->ms
        dummy =  np.zeros([edfdata.DimSize[0], 3])
        for i in range(nrange):
            edfsave = TE.edf()
            edfsave.Name     = edfdata.Name
            edfsave.ShotNo   = edfdata.ShotNo
            edfsave.SubNo    = edfdata.SubNo
            edfsave.Date     = TE.edf_formatted_date()
            edfsave.DimNo    = 1
            edfsave.DimSize  = edfdata.DimSize
            edfsave.DimName  = edfdata.DimName
            edfsave.DimUnit  = edfdata.DimUnit
            edfsave.ValNo    = 2
            edfsave.ValName  = [edfdata.ValName[target-1],'filtered']
            edfsave.ValUnit  = [edfdata.ValUnit[target-1], edfdata.ValUnit[target-1]]
            edfsave.comments = ['Converted by anahilbert']
            indx = TE.listSearch(edfdata.comments, 'Written as')
            if len(indx) > 0 :
                ipos = edfdata.comments[indx[0]].find('s')
                p =  Path(edfdata.comments[indx[0]][ipos+2:].strip(' \''))
                orgfile = p.name
                edfsave.comments.append('orgfile = {}'.format(orgfile))
            else:
                orgfile = edfdata.Name + '@' + str(edfdata.ShotNo) + '.edf'
            savefile = 'Lowpass_{}_'.format(_ynames[i].replace(' ',''))+orgfile
            edfsave.comments.append('id = {}'.format(target))
            edfsave.comments.append('dt = {} (s)'.format(dt))
            edfsave.comments.append('nord = {} '.format(args.order))
            edfsave.comments.append('fc = {} (kHz)'.format(args.cut))
            dummy[:,0] = t
            dummy[:,1] = ys
            dummy[:,2] = filtered               
            edfsave.save(dummy, fname=savefile)                      

    
