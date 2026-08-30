#!/usr/bin/env python
"""
    This program reads a formatted two-dimensional data file (edf formatted file) and converts it into a set of one-dimensional slices.
    The time-series-data should be given as "edf" format.

    Usage: jcnv2to1 [-h] [--version] -f filename [-t [id ...]] [-d dimension] [-w] [-n digits] [-i] [-I] [-m] [-p] [-r]

    Converts a formatted two-dimensional data into a set of one-dimensional slices.

    options:
        -h, --help            Show this help message and exit
        --version             Show program's version number and exit
        -f filename, --file filename
                              Set an edf-formatted file to be read
        -t [id ...], --target [id ...]
                              Set valiable IDs to be convered (default 1)
        -d dimension, --dim dimension
                              Select dimension (1 or 2)
        -w, --write           Write results to file
        -n digits, --ndigit digits
                              Number of digits after decimal point
        -i, --info            Display header information
        -I, --interactive     Interactive mode (plot graph)
        -p, --portrait        Portrait mode
        -r, --reverse         Reverse color map (default True jet_r)
    Example:
        python jcnv2to1.py -f ../194949/tsmap_smooth_a999@194949_1.txt -t 1 2 -d 1 -w -I

    Dependence
    ----------
    turnelib.py
    matplotlibutil.py
    jclass.py

    Status
    ------
    Version 1.1
    
    Revision History
    ----------------
    [12-Oct-2025] Creation                           ver 0.9
    [12-Oct-2025] Bug in save-part fixed             ver 0.9.1
    [12-Oct-2025] Released                           ver 1.0
    [07-Nov-2025] Bug in savefile fixed              ver 1.1

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
import jclass as JC

def unchosen(i):
    if i == 0 :
        return 1
    return 0

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
    version = "1.1"
    timestamp = "2025/11/07"

    myparser = TE.default_parser('jcnv2to1', 'Converts a formatted two-dimensional data into a set of one-dimensional slices.', version, timestamp)
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
        '-t',
        '--target',
        action='store',
        nargs="*",
        type=int,
        default = 1,
        metavar = 'id',
        help='Set valiable IDs to be convered (default 1)'
    )
    myparser.add_argument(
        '-d','--dim', 
        action='store',
        type=int,
        default = 1,
        metavar = 'dimension',
        help='Select dimension (1 or 2)'
    )
    myparser.add_argument(
        '-w','--write', 
        action='store_true',
        default = False,
        help='Write results to file'
    )
    myparser.add_argument(
        '-n','--ndigit', 
        action='store',
        type=int,
        default = 2,
        metavar = 'digits',
        help='Number of digits after decimal point'
    )
    myparser.add_argument(
        '-i','--info', 
        action='store_true',
        default = False,
        help='Display header information'
    )
    myparser.add_argument(
        '-I','--interactive', 
        action='store_true',
        default = False,
        help='Interactive mode (plot graph)'
    )
    myparser.add_argument(
        '-p','--portrait', 
        action='store_true',
        default = False,
        help='Portrait mode'
    )
    myparser.add_argument(
        '-m',
        '--multi',
        action='store_true',
        default = False,
        help='Multi-subplots are drawn (default False)'
    )
    myparser.add_argument(
        '-r','--reverse', 
        action='store_false',
        default = True,
        help='Reverse color map (default True jet_r)'
    )
    
    args = myparser.parse_args()
    edfdata = TE.edf()    
    dat = edfdata.load(args.file)
    path = Path(args.file)
    
    if args.info :
        id = 1
        print('Select dimension from')
        for name in edfdata.DimName:
            print(id, name)
            id = id + 1
        id = 1
        print('')
        print('Select targets from')
        for name in edfdata.ValName:
            print(id, name)
            id = id + 1
        sys.exit()
        
    target =  TE.ensure_array(args.target) + 1
    xdim = args.dim - 1
    ydim = unchosen(xdim)

    converted = []
    xx = dat[:,xdim].reshape(edfdata.DimSize)
    xunit = edfdata.ValUnit[xdim]
    yy = dat[:,ydim].reshape(edfdata.DimSize)
    yname = edfdata.DimName[ydim]
    yunit = edfdata.DimUnit[ydim]
    if xdim == 1:
        xx = xx.transpose() 
        yy = yy.transpose()
    x = xx[:,0]
    y = yy[0,:]        
    yfullnames =  valnames(y, yname, yunit, args.ndigit)
   
    for v in target:
        mat = dat[:,v].reshape(edfdata.DimSize)
        if xdim == 1:
            mat = mat.transpose()
        converted.append(mat)
        valunit = edfdata.ValUnit[v-2]

        if args.write:
            edfsave = TE.edf()
            edfsave.Name     = edfdata.ValName[v-2]
            edfsave.ShotNo   = edfdata.ShotNo
            edfsave.SubNo    = edfdata.SubNo
            edfsave.Date     = TE.edf_formatted_date()
            edfsave.DimNo    = 1
            edfsave.DimSize  = [edfdata.DimSize[xdim]]
            edfsave.DimName  = [edfdata.DimName[xdim]] 
            edfsave.DimUnit  = [edfdata.DimUnit[xdim]]
            edfsave.ValNo    = len(y)
            edfsave.ValName  = yfullnames
            edfsave.ValUnit  = [valunit]*len(y)
            edfsave.comments = copy.deepcopy(edfdata.comments)
            edfsave.comments.insert(0,'original ==>')
            edfsave.comments.append('<== original')
            edfsave.comments.append('')
            if '/' in edfsave.DimName[0]:
                savefile = edfsave.Name+'_'+edfsave.DimName[0].replace('/', '%') + '_from_' + edfdata.Name + '@' +str(edfdata.ShotNo)+ '.edf'
            else:
                savefile = edfsave.Name+'_'+edfsave.DimName[0] + '_from_' + edfdata.Name + '@' +str(edfdata.ShotNo)+ '.edf'                   
            edfsave.save(TE.to_xny(x, mat), fname=savefile) 

    if args.interactive :
        id = 1
        for fullname in yfullnames:
            print(id, fullname)
            id = id + 1
        print('Select IDs')
        ans = input('>>>')
        vals = []
        for val in  ans.split(','):
            vals.append(int(val))
        id_plot = TE.ensure_array(vals) - 1
#        selected_name = []
#        for id in id_plot :
#            selected_name.append(yfullnames[id])

        n = len(target)
        title = path.name + ' #' + str(edfdata.ShotNo)
        _xlabel =edfdata.DimName[xdim] + ' (' + edfdata.DimUnit[xdim] + ')'
        for i in range(n):
            mat = converted[i]
            _ylabel = edfdata.ValName[target[i]-2] + ' (' + edfdata.ValUnit[target[i]-2] + ')'
            JC.edfview_1D(x, mat.transpose(), _xlabel, _ylabel, yfullnames, title, id_plot=id_plot, multi=args.multi, portrait=args.portrait, reverse=args.reverse) 
            """
            page = MPU.mypage(fontsize=20, scale=0.84, style='slide', portrait=args.portrait)
            plots = page.make_subplots(1,1)
            mat = converted[i]
            _ylabel = edfdata.ValName[target[i]-2] + ' (' + edfdata.ValUnit[target[i]-2] + ')'
            plots[0].add_trace(x, mat[:,id_plot].transpose(), xlabel=_xlabel, ylabel=_ylabel, mode='lines', ynames=selected_name, colors=None)
            plots[0].legend(selected_name,loc='best')
            plots[0].set_title(title, loc='right', fontdict={'fontfamily':'monospace','fontsize':12,})
            plt.show()
            """    
 
    
