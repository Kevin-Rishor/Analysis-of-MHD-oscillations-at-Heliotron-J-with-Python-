#!/usr/bin/python
# coding: utf-8
"""
    Simple Affine Transform
    Let X and Y be n×m matrices, and let a and b be vectors of length m
    The operation
       Y=aX+b
    is defined element-wise as
       Y[i,j] = a[j]*X[i,j] + b[j]

    ------------------------------------------
    for i in range(n):
        for j in range(m):
            Y[i, j] = a[j] * X[i, j] + b[j]
    ------------------------------------------

    usage: affine.py [-h] [-f filename] -a A -b B

    options:
      -h, --help         show this help message and exit
      -f filename, --file filename
                         Set an edf-formatted file to be read
      -a A               Comma/Space-separated list (e.g. 3,4.2,5 or "3 4.2 5")
      -b B               Comma/Space-separated list (e.g. 3,4.2,5 or "3 4.2 5")

    Example:
       python affine.py -f HAFAST11.5@83848.edf -a 1,1 -b "0 0.15" 

    Dependence
    ----------
    turnelib.py

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
    [26-Dec-2025] Creation                                       ver 1.0
            
    Copyright
    ---------
    2023 Shigeru Inagaki (inagaki.shigeru.7s@kyoto-u.ac.jp)
    Released under the CC BY-NC 4.0.

"""
import argparse
from pathlib import Path
import turnelib as TE
import numpy as np

def parse_list(s, dtype=float):
    sep = ',' if ',' in s else ' '
    return np.array([dtype(x) for x in s.split(sep)])
    

if __name__== '__main__':    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-f', '--file',
        action='store',
        type=str,
        metavar='filename',
        help='Set an edf-formatted file to be read'
    )
    parser.add_argument(
        '-a',
        type=lambda s: parse_list(s, float),
        required=True,
        help='Comma/Space-separated list (e.g. 3,4.2,5 or "3 4.2 5")'
    )
    parser.add_argument(
        '-b',
        type=lambda s: parse_list(s, float),
        required=True,
        help='Comma/Space-separated list (e.g. 3,4.2,5 or "3 4.2 5")'
    )
    args = parser.parse_args()
    a = args.a
    b = args.b
    path = Path(args.file)
    savepath = path.parent / ('af_' + path.name)
    print(savepath)

    edfdata = TE.edf()
    dat = edfdata.load(args.file)
    
    dat = dat * a + b

    edfdata.comments.append('Converted by affine')
    edfdata.comments.append('a, b')
    for i in range(len(a)):
        edfdata.comments.append('{}, {}'.format(a[i], b[i]))

    edfdata.save(dat, fname=savepath)
