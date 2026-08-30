#!/usr/bin/python
# coding: utf-8
"""
    Overlay the graphs which specified by edf-formatted files

    Usage: overlay.py [-h] [-f file[=u]] [-p] [-r] [-x name] [-y name]

    options:
      -h, --help            show this help message and exit
      -f file[=u], --file filenme[=u]
                                 Specify input file.
                                 The optional parameter 'u' may be given as:
                                       n:m      denotes x-values are the n-th column (0-based)
                                                and y-values are the m-th column in the edf file.
                                        m       This is equivalent to 0:m.
                                        :m      This is equivalent to 0:m.
                                       n:       This is equivalent to n:1.
                                       If omitted, the default target 0:1 is used.
                                 This option may be specified multiple times.
      -p, --portrait        Portrait mode
      -r, --reverse         Reverse color map (default True jet_r)
      -x name, --xlabel name
                            Set X-axis label name
      -y name, --ylabel name
                            Set Y-axis label name

    Example:
       python overlay.py -f hj88841/haarr@88841.edf=0:6 -f hj88841/HAFAST15.5@88841.edf -f hj88841/haarr@88841.edf=0:12

    Dependence
    ----------
    turnelib.py
    matplotutil.py

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
    [18-Dec-2025] Creation                                       ver 1.0
            
    Copyright
    ---------
    2023 Shigeru Inagaki (inagaki.shigeru.7s@kyoto-u.ac.jp)
    Released under the CC BY-NC 4.0.

"""
import argparse
from pathlib import Path
import turnelib as TE
import matplotlib
import matplotlib.pyplot as plt
import matplotlibutil as MPU

matplotlib.axes.Axes.add_trace = MPU.add_trace
matplotlib.figure.Figure.make_subplots = MPU.make_subplots

DEFAULT_TARGET = (None, None)  

def parse_f(value):
    # file[=u]
    if '=' not in value:
        return {'file': value, 'using': DEFAULT_TARGET}

    file, u = value.split('=', 1)
    
    if ':' not in u:
        try:
            n = int(u)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid t value: '{u}'"
            )
        return {'file': file, 'using': (0, n)}

    idx, idy = u.split(':', 1)

    def parse_int_or_none(x):
        if x == '':
            return None
        return int(x)

    try:
        idx = parse_int_or_none(idx)
        idy = parse_int_or_none(idy)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid using format: '{u}'"
        )

    return {'file': file, 'using': (idx, idy)}

if __name__== '__main__':    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-f', '--file',
        action='append',
        type=parse_f,
        metavar='file[=u]',
        help='Input file with optional range u'
    )
    parser.add_argument(
        '-p','--portrait', 
        action='store_true',
        default = False,
        help='Portrait mode'
    )
    parser.add_argument(
        '-r','--reverse', 
        action='store_false',
        default = True,
        help='Reverse color map (default True jet_r)'
    )
    parser.add_argument(
        '-x',
        '--xlabel',
        action='store',
        type=str,
        default = None,
        metavar = 'name',
        help='Set X-axis label name'
    )
    parser.add_argument(
        '-y',
        '--ylabel',
        action='store',
        type=str,
        default = None,
        metavar = 'name',
        help='Set Y-axis label name'
    )
    args = parser.parse_args()

    nfile = len(args.file)
    _colors = MPU.AutoColor(nfile, reverse=args.reverse)
    page = MPU.mypage(fontsize=20, scale=0.84, style='slide', portrait=args.portrait)
    plots = page.make_subplots(1,1)
    ynames = []
    edfdata = TE.edf()
    counter = 0
    for item in args.file:
        file, (idx, idy) = item['file'], item['using']
        if idx is None:
            idx = 0
        if idy is None:
            idy = 1

        path = Path(file)
        dat = edfdata.load(file)
        x = dat[:,idx]
        y = dat[:,idy]
        if counter == nfile -1:
            if args.xlabel is None:
                if idx == 0 :
                    _xlabel = edfdata.DimName[0] + " (" + edfdata.DimUnit[0] + ")"
                else:
                    _xlabel = edfdata.ValName[idx-1] + " (" + edfdata.ValUnit[idx-1] + ")"
            else:
                _xlabel = args.xlabel
            if args.ylabel is None:
                _ylabel = edfdata.ValName[idy-1] + " (" + edfdata.ValUnit[idy-1] + ")"
            else:
                _ylabel = args.xlabel
        else:
            _ylabel = ''
            _xlabel = ''

        yname = path.stem +'({}:{})'.format(idx,idy)
        ynames.append(yname)
        plots[0].add_trace(x, y, xlabel=_xlabel, ylabel=_ylabel, mode='lines', ynames=[yname], colors=[_colors[counter]])
        counter = counter + 1
        
    leg = plots[0].legend(ynames,loc='best',frameon=False,handlelength=0,handletextpad=0.5)
    for text, color in zip(leg.get_texts(), _colors):
        text.set_color(color)
    plt.show()
