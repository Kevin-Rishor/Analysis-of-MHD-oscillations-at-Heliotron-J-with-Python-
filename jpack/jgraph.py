#!/usr/bin/env python
# coding: utf-8
"""
    This application reads a setup file made by jview.py and 
    visualizes edf-formatted files.

    Usage:
    jgraph [--version] [-j filename] [-s shotnum]

    optional arguments:
      --version             show program's version number and exit
      -j filename, --json filename
                            json-file name for initial setting
      -s shotnum, --shot shotnum
                            replace shotnum in json-file

    Example:
      jgraph.py -j myparam.json

    Dependence
    ----------
    jview.py
    turnelib.py
    matplotlibutil.py
    jclass.py

    Status
    ------
    Version 1.3

    Authour
    -------
    Shigeru Inagaki                                       
    Institute of Advanced Energy, Kyoto University
    inagaki.shigeru.7s@kyoto-u.ac.jp
    
    Revision History
    ----------------
    [06-Apr-2025] Creation                                   ver 1.0
    [05-Sep-2025] l-option added                             ver 1.1
    [08-Sep-2025] Bug in gettrace fixed                      ver 1.2
    [08-Sep-2025] jclass is used                             ver 1.3
    
    Copyright
    ---------
    2025 Shigeru Inagaki (inagaki.shigeru.7s@kyoto-u.ac.jp)
    Released under the CC BY-NC 4.0.
"""
import os
import argparse
from operator import attrgetter
import numpy as np
import jclass as JC
import turnelib as TE
import matplotlib
import matplotlib.pyplot as plt
import matplotlibutil as MPU

matplotlib.axes.Axes.add_trace = MPU.add_trace
matplotlib.axes.Axes.add_errorbar = MPU.add_errorbar
matplotlib.axes.Axes.matview = MPU.matview
matplotlib.figure.Figure.make_subplots = MPU.make_subplots

DIRSEP = os.sep

def fullname(datapath, pref, name, strshot, ext, opt_l):
    if opt_l:
       return TE.checksep(datapath) + name
    else:
       dir = pref + strshot
       return TE.checksep(datapath) + dir + DIRSEP + name + '@' + strshot + '.' + ext         

def gettrace(trace, file, xunit):
    dataloader = TE.edf()
    if trace.nskip > 1 :
       org = dataloader.load(fname=file)
       dat = org[::trace.nskip,:]
    else:
       dat = dataloader.load(fname=file)

    if xunit == 'ms':
        if dataloader.DimUnit[0] == 's':
            dat[:,0] = 1.0e3*dat[:,0]
    elif xunit == 's':
        if dataloader.DimUnit[0] == 'ms':
            dat[:,0] = dat[:,0]/1.0e3
            
    if trace.xmask :
        xindx = np.where(np.logical_and(dat[:,0] >= trace.xmin, dat[:,0] <=trace.xmax))

    if trace.vals == 'all' or trace.vals == 'ALL' or trace.vals == 'All':
        if trace.xmask:
            time = dat[xindx[0],0]
            ys = dat[xindx[0],1:]
        else:
            time = dat[:,0]
            ys = dat[:,1:]
        legend = dataloader.ValName
    else:
        vals = np.array(JC.split2int(trace.vals))
        if trace.xmask:
            time = dat[xindx[0],0]
            dummy = dat[xindx[0],:]
            ys = dummy[:,vals]
#            ys = dat[xindx[0],vals]
        else:
            time = dat[:,0]
            ys = dat[:,vals]
        if len(vals) > 1 :
            if trace.tags == 'use_ValName':
                legend = []
                for i in vals:
                    legend.append(dataloader.ValName[i-1])
            else:
                legend = trace.tags.split()
        else:
            if trace.tags == 'use_shotnum':
                legend = [trace.strshot]
            elif trace.tags == 'use_name':
                legend = [trace.name]
            else:
                legend = [trace.tags]       

    ys = trace.scale * ys + trace.offset
    
    if trace.color == 'auto':
        colors = None
    else:
        n = len(ys[0,:])
        colors = [trace.color]*n
    return time, ys.transpose(), legend, colors

def gettitle(view, trace):
    if view.autotitle:
        if view.multishot :
            title = view.device
            shots = []
            for t in view.trace:
                shots.append(t.strshot)
            unique_shots = list(set(shots))
            unique_shots.sort(key=int)
            for strshot in unique_shots:
                title = title + ' #' + strshot
        else:
            dir = view.dprf + trace.strshot
            fname = view.datapath + DIRSEP + dir + DIRSEP + trace.strshot + '.' + 'info'       
            with open(fname, 'r') as f:
                lines = f.readlines()
            title = view.device + '  #' + lines[0].strip()
    else:
        title = view.title
        title.replace('{device}',view.device)
        title.replace('{shot}',str(view.shotnum))
    return title
    
            
if __name__== '__main__':
    parser = argparse.ArgumentParser(
        prog='jgraph',
        usage='jgraph --json tmp.json',
        description='Visualization of edf-formatted files',
        add_help=True,
    )
    parser.add_argument(
        '--version', 
        action='version', 
        version='Ver1.0'
    )
    parser.add_argument(
        '-j',
        '--json',
        action='store',
        type=str,
        default = None,
        metavar = 'jsonfile',
        help='setup json file created by jview'
    )
    parser.add_argument(
        '-s',
        '--shot',
        action='store',
        type=str,
        default = None,
        metavar = 'shot_number',
        help='overwrite shotnum'
    )
    parser.add_argument(
        '-l',
        '--local',
        action='store_true',
        default = False,
        help='Read local files'
    ) 
    args = parser.parse_args()
    
    title = None
    setupfile = args.json
    view = JC.loadfromjson(setupfile)

# Overwrite shot number

    if not view.multishot :       
        if args.shot is not None:
            strshot = args.shot
        else:
            strshot = str(view.shotnum)
        for t in view.traces:
            t.strshot = strshot

# Check and Sort the frame

    dummy = []
    for f in view.frames:
        if f.draw :
            dummy.append(f)
    view.frames = sorted(dummy, key=attrgetter('frameid'))
    maxframe = len(view.frames) 

# setup plot from frame[0]

    nres = view.ncol*view.nrow - maxframe

    if view.fontsize == 'auto':
        if view.pagestyle == 'slide' :
            if view.nrow > 6 :
                _fontsize = 12
            else:
                _fontsize = 20
        else:
            if view.nrow > 6 :
                _fontsize = 8
            else:
                _fontsize = 14
    else:
        _fontsize = int(view.fontsize)
        
    if view.orientation == 'portrait':
        _portrait = True
        _scale = 0.84
    else:
        _portrait = False
        _scale = 1.2

    page = MPU.mypage(fontsize=_fontsize, scale=_scale, style=view.pagestyle, portrait=_portrait)

    if view.autospace :
        if view.pagestyle == 'slide' :
            _wspace = 0.25
        else :
            _wspace = 0.2
        _hspace = 0.1
    else :
        _wspace = view.wspace
        _hspace = view.hspace
            
    if view.automargin :
        _left = 0.1
        _right=0.95
        _top=0.9
        _bottom = 0.1
    else :
        _left = view.left
        _right = 1.0-view.right
        _top = 1.0-view.top
        _bottom = view.bottom
    plots = page.make_subplots(view.nrow, view.ncol, left=_left, right=_right, top=_top, bottom=_bottom, wspace=_wspace, hspace=_hspace)


    opt_local = not(view.jview) or args.local

    if view.nrow == 1 :
        legend = []
        for obj in view.traces:
            if obj.frame == 1 and obj.visible :
                fname = fullname(view.datapath, view.dprf, obj.name, obj.strshot, view.fext, opt_local)
                x, ys, ylegend, colors = gettrace(obj, fname, view.xunit)
                plots[0].add_trace(x, ys, xlabel=view.frames[0].xname, ylabel=view.frames[0].yname, mode='lines', colors=colors, ynames=legend)
                if isinstance(ylegend, list):
                    legend += ylegend
                else:
                    legend.append(ylegend) 
                if title is None:
                    title =  gettitle(view, obj)                   
        if view.frames[0].xrangefix :
            plots[0].set_xlim([view.frames[0].xmin, view.frames[0].xmax])
        if view.frames[0].yrangefix :
            plots[0].set_ylim([view.frames[0].ymin, view.frames[0].ymax])
        if len(legend) > 1 and view.frames[0].legend :
            plots[0].legend(legend,loc='best')
        plots[0].set_title(title, loc='right', fontdict={'fontfamily':'monospace','fontsize':12,})
    else :
        k = 0
        for j in range(view.ncol) :
            for i in range(view.nrow) :
                if k > maxframe -1 :
                    break
                legend = []                
                if i < view.nrow - 1 and k < maxframe - 1:   
                    for obj in view.traces:                  
                        if obj.frame == k + 1 and obj.visible :
                            fname = fullname(view.datapath, view.dprf, obj.name, obj.strshot, view.fext, opt_local)
                            x, ys, ylegend, colors = gettrace(obj, fname, view.xunit)
                            plots[k].add_trace(x, ys, xlabel='', ylabel=view.frames[k].yname, mode='lines', colors=colors, ynames=ylegend)
                            if isinstance(ylegend, list):
                                legend += ylegend
                            else:
                                legend.append(ylegend)
                            if title is None:
                                title =  gettitle(view, obj)    
                    plt.setp(plots[k].get_xticklabels(), visible=False)  
                else :
                    for obj in view.traces:
                        if obj.frame == k + 1 and obj.visible :
                            fname = fullname(view.datapath, view.dprf, obj.name, obj.strshot, view.fext, opt_local)
                            x, ys, ylegend, colors = gettrace(obj, fname, view.xunit)
                            plots[k].add_trace(x, ys, xlabel=view.frames[k].xname, ylabel=view.frames[k].yname, mode='lines', colors=colors, ynames=ylegend)
                            if isinstance(ylegend, list):
                                legend += ylegend
                            else:
                                legend.append(ylegend)
                            if title is None:
                                title =  gettitle(view, obj)    
                if len(legend) > 1 and view.frames[k].legend : 
                    plots[k].legend(legend,loc='best')
                if view.frames[k].xrangefix :
                    plots[k].set_xlim([view.frames[k].xmin, view.frames[k].xmax])
                if view.frames[k].yrangefix :
                    plots[k].set_ylim([view.frames[k].ymin, view.frames[k].ymax])   
                k = k + 1
        if nres > 0 :
            for j in range(nres):
                plots[j+maxframe].axis('off')
        page.suptitle(title, fontsize=18, y=_top + 0.05)
    plt.show()
