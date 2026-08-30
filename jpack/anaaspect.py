#!/usr/bin/env python
"""
    This application calculates auto power spectrum of a time-series-data.
    The time-series-data should be given as "edf" format.
    Outputs are results of
         1. running FFT (short time FFT)
         2. Auto Power Spectrum which is ensemble averaged over a specified time period.

    Usage:
    anaaspect [-h] [--version] [-f filename] [-t id] -r fres [-w]

    optional arguments:
        -h, --help             Show this help message and exit
        --version              Show program's version number and exit
        -f filename, --file filename
                              Set an edf-formatted file to be read
        -t id, --target id      Set variables IDs to be plotted (default 1)
        -r fres, --resolution fres
                              Set frequency resolution in [Hz]
        -w, --write            Save results to files

    Example:
        anaaspect.py -f ../expdata/FY2022/MP3@83489.edf -r 500

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
    [29-Nov-2022] Creation                              ver 0.9.3
    [06-Oct-2023] Revised                               ver 0.9.4
    [06-Oct-2023] Bug in save file fixed                 ver 0.9.5
    [12-Nov-2025] -n option removed, -w option added       ver 1.0

    Copyright
    ---------
    2023 Shigeru Inagaki (inagaki.shigeru.7s@kyoto-u.ac.jp)
    Released under the CC BY-NC 4.0.

"""

import os
import math
from pathlib import Path
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

def submit(text):
    ydata = eval(text)
    l.set_ydata(ydata)
    ax.set_ylim(np.min(ydata), np.max(ydata))
    plt.draw()

my_round_int = lambda x: int((x * 2 + 1) // 2)

if __name__ == "__main__":
    version = "1.0"
    timestamp = "2025/11/12"

    myparser = TE.default_parser('anaaspect', 'Spectrum analysis (Auto-Spectrum)', version, timestamp)
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
        '-r',
        '--resolution',
        action='store',
        type=float,
        default = None,
        required = True,
        metavar = 'fres',
        help='Set frequency resolution in [Hz]'
    )
    myparser.add_argument(
        '-w',
        '--write',
        action='store_true',
        default = False,
        help='Save results to files'
    ) 
    args = myparser.parse_args()
    
    edfdata = TE.edf()    
    if args.file is None :
        dat = edfdata.load()
        title = '#' + edfdata.Name + '@' + str(edfdata.ShotNo)
    else:
        dat = edfdata.load(args.file)
        p = Path(args.file)
        title = p.stem
    target = args.target 
#    fres = 500.0 # Hz
    fres = args.resolution
    
    t = dat[:,0]
    ys = dat[:,target]

    if edfdata.DimUnit[0] == 'ms' :
        t = t/1000.0
    dt = (t[100]-t[0])/100
   # print(dt)

    page = MPU.mypage()
    plots = page.make_subplots(1,1,  bottom=0.4)
    plots[0].add_trace(t*1000, ys, xlabel='t (ms)', ylabel='Output (V)', mode='lines')

    tranges = []
    inputs = []
    def SetRanges(event):
        global tranges, inputs, vline1, vline2
        text = text_box.text
        d = text.split(',')
        s = [d[0].strip(),d[1].strip()]
        v = [float(d[0]), float(d[1])]
        inputs.append(s)
        tranges.append(v)
        ymin, ymax = plots[0].get_ylim()
        vline1 = plots[0].vlines(v[0],ymin,ymax,linestyle='dashed')
        vline2 = plots[0].vlines(v[1],ymin,ymax,linestyle='dashed')
        return

    def Reset(event):
        vline1.remove()
        vline2.remove()
        last = tranges.pop()
        return
    
    def MoveOn(event):
        plt.close()
        return

    axbox = plt.axes([0.4, 0.2, 0.2, 0.06])
    text_box = TextBox(axbox, 't1, t2 [ms] ? ', initial='200, 300')
    axbutton_set = plt.axes([0.15, 0.1, 0.1, 0.06])
    button_set = Button(axbutton_set, 'Set')
    button_set.on_clicked(SetRanges)
    axbutton_reset = plt.axes([0.26, 0.1, 0.1, 0.06])
    button_reset = Button(axbutton_reset, 'Reset')
    button_reset.on_clicked(Reset)
    axbutton_ok = plt.axes([0.37, 0.1, 0.1, 0.06])
    button_ok = Button(axbutton_ok, 'OK')
    button_ok.on_clicked(MoveOn)  
    plt.show()
    print('Times of interest:')
    print(tranges)
    
    nfft  = my_round_int(1/(dt*fres))
    f, tave, Pyy = LAS.running(ys, t, dt=dt, nfft=nfft, noverlap = nfft//2, detrend='constant')
    tt, ff =  np.meshgrid(tave,f)

# --- save stfft ---
    if args.write :
        edfsave = TE.edf()
        edfsave.Name     = edfdata.Name
        edfsave.ShotNo   = edfdata.ShotNo
        edfsave.SubNo    = edfdata.SubNo
        edfsave.Date     = TE.edf_formatted_date()
        edfsave.DimNo    = 2
        edfsave.DimSize  = [len(tave),len(f)]
        edfsave.DimName  = ['time', 'freq']
        edfsave.DimUnit  = ['s', 'Hz']
        edfsave.ValNo    = 1
        edfsave.ValName  = ['PSD of '+edfdata.ValName[target-1]]
        edfsave.ValUnit  = [edfdata.ValUnit[target-1]+'^2/Hz']
        edfsave.comments = ['Converted by anaaspct']
        indx = TE.listSearch(edfdata.comments, 'Written as')
        if len(indx) > 0 :
            edfsave.comments.append(edfdata.comments[indx[0]])
            i = edfdata.comments[indx[0]].find('s')
            p =  Path(edfdata.comments[indx[0]][i+2:].strip(' \''))
            orgfile = p.name
        else:
            orgfile = edfdata.Name + '@' + str(edfdata.ShotNo) + '.edf'
        savefile = 'stfft_'+orgfile
        edfsave.comments.append('id = {}'.format(target))
        edfsave.comments.append('nfft = {}'.format(nfft))
        edfsave.comments.append('fres = {} (Hz)'.format(fres))
        edfsave.comments.append('dt = {} (s)'.format(dt))
        edfsave.save(TE.to_xyz(tave, f, Pyy.transpose()), fname=savefile)
        
    page = MPU.mypage()
    plots = page.make_subplots(1,1)
    Pmax = TE.maxval(Pyy)
    Pmin = TE.minval(Pyy)
    cont = plots[0].contourf(tt*1000, ff/1000, np.log10(Pyy), cmap='jet',)
    cb = plt.colorbar(cont)   #ticks=[0.0001,0.001,0.01,0.1,1])
    cb.set_label('log10 PSD (V**2/Hz)')  
    plots[0].set_ylabel('f (kHz)')
    plots[0].set_xlabel('t (ms)')
    plots[0].set_title(title, loc='right', fontdict={'fontfamily':'monospace','fontsize':12,})
    ymin, ymax = plots[0].get_ylim()
    for t12 in tranges:
         plots[0].vlines(t12[0],ymin,ymax,linestyle='dashed')
         plots[0].vlines(t12[1],ymin,ymax,linestyle='dashed')    
    plt.show()

    _ynames = []
    nrange = len(tranges)
    spct = np.zeros([nrange,len(f)])
    for i in range(nrange):
        name = '{}-{} ms'.format(inputs[i][0],inputs[i][1])
        _ynames.append(name)
        tindx =  np.where(np.logical_and(tave > tranges[i][0]/1000, tave<tranges[i][1]/1000))
        spct[i,:] = np.average(Pyy[:,tindx[0]], axis=1)

# --- save spct ---
    if args.write :
        edfsave = TE.edf()
        edfsave.Name     = edfdata.Name
        edfsave.ShotNo   = edfdata.ShotNo
        edfsave.SubNo    = edfdata.SubNo
        edfsave.Date     = TE.edf_formatted_date()
        edfsave.DimNo    = 1
        edfsave.DimSize  = [len(f)]
        edfsave.DimName  = ['freq']
        edfsave.DimUnit  = ['Hz']
        edfsave.ValNo    = nrange
        edfsave.ValName  = _ynames
        edfsave.ValUnit  = [edfdata.ValUnit[target-1]+'^2/Hz']*nrange
        edfsave.comments = ['Converted by anaaspct']
        indx = TE.listSearch(edfdata.comments, 'Written as')
        if len(indx) > 0 :
            i = edfdata.comments[indx[0]].find('s')
            p =  Path(edfdata.comments[indx[0]][i+2:].strip(' \''))
            orgfile = p.name
            edfsave.comments.append('orgfile = {}'.format(orgfile))
        else:
            orgfile = edfdata.Name + '@' + str(edfdata.ShotNo) + '.edf'
        savefile = 'spct_'+orgfile
        edfsave.comments.append('id = {}'.format(target))
        edfsave.comments.append('nfft = {}'.format(nfft))
        edfsave.comments.append('fres = {} (Hz)'.format(fres))
        edfsave.comments.append('dt = {} (s)'.format(dt))
        edfsave.save(TE.to_xny(f, spct.transpose()), fname=savefile)

    page = MPU.mypage()
    plots = page.make_subplots(1,1)
#    plots[0].add_trace(f/1000, TE.to_xny(spct1, spct2).transpose(), xlabel='f (kHz)', ylabel='PSD (V2/Hz)', mode='lines', ynames=['280-300 ms', '310-330 ms'])
    plots[0].add_trace(f/1000, spct, xlabel='f (kHz)', ylabel='PSD (V2/Hz)', mode='lines', ynames=_ynames)
    plots[0].set_title(title, loc='right', fontdict={'fontfamily':'monospace','fontsize':12,})
    plt.show()

