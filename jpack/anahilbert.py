#!/usr/bin/env python
"""
    This application calculates analysis signal of a time-series-data.
    The time-series-data should be given as "edf" format.
    Outputs are results of
         1. Band-passed signal
         2. Envelope of the band-passed signal
         3. Instantaneous frequency which is calculated from time-derivative of
            the instantaneous phase

    Usage:
    anahilbert [-h] [--version] [-f filename] [-t id] [-s sgwin] [-l fl]
                    [-u fu] [-o nord] [-w]


    optional arguments:
        -h, --help             show this help message and exit
        --version              show program's version number and exit
        -f filename, --file filename
                               Set an edf-formatted file to be read
        -t id, --target id     Set variables IDs to be plotted (default 1)
        -s sgwin, --smoothing sgwin
                               Set window width for savgol_filter (odd-integer)
        -l fl, --lower fl      Set lower frequency for band-pass-filter in [kHz]
        -u fu, --upper fu      Set upper frequency for band-pass-filter in [kHz]
        -o nord, --order nord
                               Set order for Bessel filter (default=4)
        -w, --write            Save results to files

    Example:
        python anahilbert.py -f ../expdata/FY2022/MP3@83489.edf 

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

def Hilbert_Envelope(xin, fsamp, fl, fu, nord=4):
    twopi = 2*np.pi
    omega_l = twopi*fl
    omega_u = twopi*fu
    A_numerator, A_denominator =  dsp.bessel(nord, [omega_l, omega_u], 'bandpass', analog=True)
    D_numerator, D_denominator = dsp.bilinear(A_numerator, A_denominator, fsamp)
    filtered = np.zeros(len(xin))
    envelope = np.zeros(len(xin))
    phase = np.zeros(len(xin))
    filtered = dsp.filtfilt(D_numerator, D_denominator, xin)       
    analytic = dsp.hilbert(filtered)
    envelope = np.abs(analytic)
    A_numerator, A_denominator =  dsp.bessel(nord, omega_l/2, 'lowpass', analog=True)
    D_numerator, D_denominator = dsp.bilinear(A_numerator, A_denominator, fsamp)
    envelope = dsp.filtfilt(D_numerator, D_denominator, envelope)
    phase = np.unwrap(np.angle(analytic))/twopi
    return envelope, phase, filtered
    
def submit(text):
    ydata = eval(text)
    l.set_ydata(ydata)
    ax.set_ylim(np.min(ydata), np.max(ydata))
    plt.draw()

my_round_int = lambda x: int((x * 2 + 1) // 2)

if __name__ == "__main__":
    version = "1.0"
    timestamp = "2025/11/12"

    myparser = TE.default_parser('anahilbert', 'Time-series data analysis by using the Hilbert transform', version, timestamp)
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
        '-s',
        '--smoothing',
        action='store',
        type=int,
        default = 525,
        metavar = 'sgwin',
        help='Set window width for savgol_filter (odd-integer)'
    )
    myparser.add_argument(
        '-l',
        '--lower',
        action='store',
        type=float,
        default = None,
        metavar = 'fl',
        help='Set lower frequency for band-pass-filter in [kHz]'
    )
    myparser.add_argument(
        '-u',
        '--upper',
        action='store',
        type=float,
        default = None,
        metavar = 'fu',
        help='Set upper frequency for band-pass-filter in [kHz]'
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
#    fres = args.resolution
    fres = 500.0

    t = dat[:,0]
    ys = dat[:,target]
    if edfdata.ValNo == 1 :
        _ylabel = edfdata.Name + '({})'.format(edfdata.ValUnit[target-1])
    else :
        _ylabel = edfdata.ValName[target-1] + '({})'.format(edfdata.ValUnit[target-1])

    if edfdata.DimUnit[0] == 'ms' :
        t = t/1000.0  #ms-->sec
    dt = (t[100]-t[0])/100

#--- page 1 ---
    page = MPU.mypage()
    plots = page.make_subplots(1,1)
    plots[0].add_trace(t*1000, ys, xlabel='t (ms)', ylabel=_ylabel, mode='lines')
    plots[0].set_title(title, loc='right', fontdict={'fontfamily':'monospace','fontsize':12,})
    plt.show()

    franges = []
    inputs = []
    def SetRanges(event):
        global franges, inputs, hline1, hline2
        text = text_box.text
        d = text.split(',')
        s = [d[0].strip(),d[1].strip()]
        v = [float(d[0]), float(d[1])]
        inputs.append(s)
        franges.append(v)
        xmin, xmax = plots[0].get_xlim()
        hline1 = plots[0].hlines(v[0],xmin,xmax,linestyle='dashed')
        hline2 = plots[0].hlines(v[1],xmin,xmax,linestyle='dashed')
        return

    def Reset(event):
        hline1.remove()
        hline2.remove()
        last = franges.pop()
        return
    
    def MoveOn(event):
        plt.close()
        return
    
    
    if args.lower is None or args.upper is None :
        
#--- Running FFT ---
        nfft  = my_round_int(1/(dt*fres))
        f, tave, Pyy = LAS.running(ys, t, dt=dt, nfft=nfft, noverlap = nfft//2, detrend='constant')
        tt, ff =  np.meshgrid(tave,f)

#--- page 2 ---
        page = MPU.mypage()
        plots = page.make_subplots(1,1, bottom=0.4)
        Pmax = TE.maxval(Pyy)
        Pmin = TE.minval(Pyy)
        cont = plots[0].contourf(tt*1000, ff/1000, np.log10(Pyy), cmap='jet',)
        cb = plt.colorbar(cont)   #ticks=[0.0001,0.001,0.01,0.1,1])
        cb.set_label('log10 PSD (V**2/Hz)')  
        plots[0].set_ylabel('f (kHz)')
        plots[0].set_xlabel('t (ms)')
        plots[0].set_title(title, loc='right', fontdict={'fontfamily':'monospace','fontsize':12,})    

        axbox = plt.axes([0.4, 0.2, 0.2, 0.06])
        text_box = TextBox(axbox, 'f1, f2 [kHz] ? ', initial='10, 100')
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
        print('Freqs of interest:')
        print(franges)
    else:
        franges.append([args.lower,args.upper])
        inputs.append([int(args.lower),int(args.upper)])
        
#--- Envelope ---
    _ynames = []
    nrange = len(franges)
    filtered = np.zeros([nrange,edfdata.DimSize[0]])
    envelope = np.zeros([nrange,edfdata.DimSize[0]])
    phase = np.zeros([nrange,edfdata.DimSize[0]])
    ifreq =  np.zeros([nrange,edfdata.DimSize[0]])
    
    kHz = 1.0e3

    page = MPU.mypage()
    plots = page.make_subplots(nrange,1)    
    for i in range(nrange):
        name = '{}-{} kHz'.format(inputs[i][0],inputs[i][1])
        _ynames.append(name)
        envelope[i,:], phase[i,:], filtered[i,:] = Hilbert_Envelope(ys, 1/dt, franges[i][0]*kHz, franges[i][1]*kHz,args.order)
        plots[i].plot(t*1000, filtered[i,:], label='filt', color='k')
        plots[i].plot(t*1000, envelope[i,:], label='env', color='r')
        plots[i].set_ylabel(_ynames[i])
        plots[i].legend(loc='best')
#        plots[0].add_trace(t*1000, envelope[i,*], xlabel='t (ms)', ylabel=_ynames[i], mode='lines', ynames=['env'])
        if i != nrange - 1 :
            plt.setp(plots[i].get_xticklabels(), visible=False)
        else:
            plots[i].set_xlabel('Time (ms)')
    plots[0].set_title(title, loc='right', fontdict={'fontfamily':'monospace','fontsize':12,})
    plt.show()

    page = MPU.mypage()
    plots = page.make_subplots(nrange,1)
#    sg_win = 525  #125
    sg_win = args.smoothing
    for i in range(nrange):
        sphase= dsp.savgol_filter(phase[i,:], sg_win, 2)
        ifreq[i,:] = dsp.savgol_filter(phase[i,:], sg_win, 2, deriv=1)/dt
        plots[i].plot(t*1000, ifreq[i,:]/1000, label='freq', color='r')
#        plots[i].plot(t*1000, phase[i,:]-sphase, label='phase', color='k')
        plots[i].set_ylabel(_ynames[i])
        plots[i].legend(loc='best')
        if i != nrange - 1 :
            plt.setp(plots[i].get_xticklabels(), visible=False)
        else:
            plots[i].set_xlabel('Time (ms)')
    plots[0].set_title(title, loc='right', fontdict={'fontfamily':'monospace','fontsize':12,})
    plt.show()


    if args.write :
        if edfdata.DimUnit[0] == 'ms' :
            t = t*1000.0  #sec-->ms
        dummy =  np.zeros([edfdata.DimSize[0], 6])
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
            edfsave.ValNo    = 5
            edfsave.ValName  = [edfdata.ValName[target-1],'filtered','env','phase/2pi','freq']
            edfsave.ValUnit  = [edfdata.ValUnit[target-1], edfdata.ValUnit[target-1],edfdata.ValUnit[target-1],'','kHz']
            edfsave.comments = ['Converted by anahilbert']
            indx = TE.listSearch(edfdata.comments, 'Written as')
            if len(indx) > 0 :
                ipos = edfdata.comments[indx[0]].find('s')
                p =  Path(edfdata.comments[indx[0]][ipos+2:].strip(' \''))
                orgfile = p.name
                edfsave.comments.append('orgfile = {}'.format(orgfile))
            else:
                orgfile = edfdata.Name + '@' + str(edfdata.ShotNo) + '.edf'
            savefile = 'Hilbert_{}_'.format(_ynames[i].replace(' ',''))+orgfile
            edfsave.comments.append('id = {}'.format(target))
            edfsave.comments.append('dt = {} (s)'.format(dt))
            edfsave.comments.append('nord = {} '.format(args.order))
            edfsave.comments.append('fl = {} (kHz)'.format(franges[i][0]))
            edfsave.comments.append('fu = {} (kHz)'.format(franges[i][1]))
            edfsave.comments.append('sgwin = {} '.format(sg_win))
            dummy[:,0] = t
            dummy[:,1] = ys
            dummy[:,2] = filtered[i,:]
            dummy[:,3] = envelope[i,:]
            dummy[:,4] = phase[i,:]
            dummy[:,5] =  ifreq[i,:]/1000                
            edfsave.save(dummy, fname=savefile)                      

    
