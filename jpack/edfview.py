#!/usr/bin/python
# coding: utf-8
"""
    Visualize an edf-format file by using matplotlib

    Usage:
    edfview [-h] [--version] [-f filename] [-s num] [-d name] [-t id [id ...]] [-y name] [-m] [-p] [-r] [-w]

    options:
      -h, --help            Show this help message and exit
      --version             Show program's version number and exit
      -f filename, --file filename
                            Set an edf-formatted file to be read
      -s num, --shot num    Set a shot number (LHD mode only)
      -d name, --diagname name
                            Set a diagname (LHD mode only)
      -t id [id ...], --target id [id ...]
                            Set variables IDs to be plotted (default 1)
      -y name, --ylabel name
                            Set Y-axis label name
      -m, --multi           Multi-subplots are drawn (default False)
      -p, --portrait        Portrait mode
      -r, --reverse         Reverse color map default (default True jet_r)
      -w, --write           Save data (LHD mode only)

    Example:
      python edfview.py -f Te_fit_Time_from_tsmap_smooth_a999@194949.edf -t 1 11 21 31 41 51 61 71 81 91 -y '$n_\mathrm{e}\ (10^{19}\ \mathrm{m}^{-3})$' -m
      other_prog | edfview.py -t 1 2 4
      python edfview.py -s 194949 -d ech -m -w

    Dependence
    ----------
    turnelib.py
    matplotutil.py
    jclass.py
    jslang.py

    Status
    ------
    Version 2.0.1

    Authour
    -------
    Shigeru Inagaki                                       
    Institute of Advanced Energy, Kyoto University
    inagaki.shigeru.7s@kyoto-u.ac.jp
    
    Revision History
    ----------------
    [26-Oct-2022] Creation                                 ver 0.9.8
    [04-Oct-2023] Bug fixed                                ver 0.9.9
    [13-Oct-2023] Bug in args.target fixed                 ver 1.0
    [14-Oct-2025] edfview_1d added                         ver 1.1
    [19-Oct-2025] Support for igetfile (LHD mode) added    ver 2.0
    [19-Oct-2025] Bug in npz interface fixed               ver 2.0.1
            
    Copyright
    ---------
    2023 Shigeru Inagaki (inagaki.shigeru.7s@kyoto-u.ac.jp)
    Released under the CC BY-NC 4.0.

"""
from pathlib import Path
import numpy as np
import turnelib as TE
import jclass as JC
import jslang as JS
#import hjlib as HJ

TE.edf.igetfile = JS.igetfile
#TE.edf.igetfile = HJ.igetfile

            
if __name__== '__main__':    
    version = "2.0"
    timestamp = "2025/10/19"

    myparser = TE.default_parser('edfview', 'Visualize an edf-format file by matplotlib', version, timestamp)
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
        '-s',
        '--shot',
        action='store',
        type=int,
        default = None,
        metavar = 'num',
        help='Set a shot number (LHD mode)'
    )
    myparser.add_argument(
        '-d',
        '--diagname',
        action='store',
        type=str,
        default = None,
        metavar = 'name',
        help='Set a diagname (LHD mode)'
    )
    myparser.add_argument(
        '-t',
        '--target',
        action='store',
        nargs='+',
        type=int,
        default = None,
        metavar = 'id',
        help='Set variables IDs to be plotted (default 1)'
    )
    myparser.add_argument(
        '-y',
        '--ylabel',
        action='store',
        type=str,
        default = None,
        metavar = 'name',
        help='Set Y-axis label name'
    )
    myparser.add_argument(
        '-m',
        '--multi',
        action='store_true',
        default = False,
        help='Multi-subplots are drawn (default False)'
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
        '-w','--write', 
        action='store_true',
        default = False,
        help='Save data (LHD mode)'
    )
    args = myparser.parse_args()

    fromnpz = False
    if args.file is not None:
        if args.file[-3:] == 'npz' :
            fromnpz = True
    
    if fromnpz :
        npz = np.load(args.file)
        x = npz['x']
        y = npz['y']
        graph = npz['graph']
        xlabel = graph[0]
        ylabel = graph[1]
        title = graph[2]
        ylabels = [ylabel]
        ynames = ylabels
        ys = y.reshape([1,len(y)])
    else :
        edfdata = TE.edf()    
        if args.file is None :
            if (args.shot is not None) and (args.diagname is not None) :
                dat = edfdata.igetfile(args.diagname, args.shot)
                if args.write :
                    savefile = args.diagname + '@' + str(args.shot) + '.edf'
                    edfdata.save(dat, fname=savefile)
            else:    
                dat = edfdata.load()
            title = '#' + edfdata.Name + '@' + str(edfdata.ShotNo)
        else:
            path = Path(args.file)
            dat = edfdata.load(args.file)
            title = path.name + ' #' + str(edfdata.ShotNo)
        x = dat[:,0]
        ys = dat[:,1:].transpose()
        xlabel = edfdata.DimName[0] + " (" + edfdata.DimUnit[0] + ")"
        ynames = edfdata.ValName
            
    if args.target is not None:
        indx = np.array(args.target, dtype=int) - 1
    else:
        indx = TE.indgen(len(ys[:,0]))
        
    if args.ylabel is None:
        if not fromnpz :
            ylabel = edfdata.ValName[indx[0]] 
    else:
        ylabel = args.ylabel        
        
    JC.edfview_1D(x, ys, xlabel, ylabel, ynames, title, id_plot = indx, multi=args.multi, portrait=args.portrait, reverse=args.reverse)    
