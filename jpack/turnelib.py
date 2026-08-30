#!/usr/bin/env python
"""
    Utility tools

    fortran compatibility:
       findloc
       achar
       iachar
       count
       adjustl
       adjustr
       trim
       len_trim
       index
       scan
       repeat
       verify
       maxloc
       minloc
       maxval
       minval
       pack

    OS utility
       delfile
       dirscan
       seekdefault

    original:
       indgen
       findgen
       indmask
       firstloc
       replace
       btrim
       strzfill
       str2line

       strarr
       getlog
       loadtxt
       savetxt
       edf_formatted_date
       line2list
       listSearch
       smart_split
       list2float
       list2int
       select_elements
       ensure_array

    class:
       edf

    extension:
       setDefaultArguments
       setFileArguments

    Dependence
    ----------

    Status
    ------
    Version 1.2

    Authour
    -------
    Shigeru Inagaki                                       
    Institute of Advanced Energy, Kyoto University
    inagaki.shigeru.7s@kyoto-u.ac.jp
    
    Revision History
    ----------------
    [01-Aug-2022] Creation                                        Ver 0.9.0
    [21-Oct-2022] dirscan, seekdefault added                      Ver 0.9.1
    [28-Oct-2022] xmlindent, show, checksep str2list
                  added                                           Ver 0.9.2
    [17-Dec-2022] Bug in firstloc fixed                           Ver 0.9.3
    [26-Sep-2023] seekdefault revised                             Ver 0.9.4
    [03-Oct-2023] Bug in edf fixed                                Ver 0.9.5
    [03-Oct-2023] edf_formatted_date
                  listSearch  added                               Ver 0.9.6
    [09-Jan-2025] Default precision for edf.save
                  changed  %.6e                                   Ver 1.0.0
    [12-Oct-2025] Bugs in loadtxt fixed                           Ver 1.1
    [12-Oct-2025] Bugs in edf class fixed                         Ver 1.1.1
    [12-Oct-2025] smart_split, list2float, list2int,               Ver 1.2
                           select_elements added

    Copyright
    ---------
    2023 Shigeru Inagaki (inagaki.shigeru.7s@kyoto-u.ac.jp)
    Released under the CC BY-NC 4.0.


"""
import argparse
import sys
import glob
import os
import re
import itertools
import warnings
import argparse
from pathlib import Path
from time import gmtime, strftime, localtime
import numpy as np
def asbytes(s):
    if isinstance(s, bytes):
        return s
    return str(s).encode('latin1')
import xml.etree.ElementTree as ET


#------- Fortran Compatible -------
def findloc(array, value, back=False):
    if isinstance(array, list) :
        array = np.array(array)
    mask = (array == value)
    indx = np.where(mask)
    n = mask.ndim
    if back :
        i = -1
    else :
        i = 0
    if n == 1 :
        return indx[:][i]
    else :
        ans = []
        for j in range(n):
            ans.append(indx[j][i])
        return np.array(ans)

def achar(indx):
    return chr(indx)

def iachar(char):
    return ord(char)

def count(mask):
    if isinstance(mask,np.ndarray) :
        return np.sum(mask)
    else:
        return np.sum(np.ravel(mask))

def adjustl(string):
    nblank = 0
    while True :
        if (string[nblank] == ' '):
            nblank = nblank + 1
        else:
            break
        if nblank == len(string) :
            return string
    return string[nblank:]+' '*nblank    

def adjustr(string):
    dummy = string[::-1]
    dummy = adjustl(dummy)
    return dummy[::-1]

def trim(string):
    return string.rstrip()

def len_trim(string):
    return len(trim(string))

def index(string, target):
    return string.find(target)

def scan(string, set, back=False):
    if len(set) == 0 :
        return -1
    if back :
        for char in set[-1]:
            indx = string.find(char)
            if indx != -1 :
                break
    else:
        for char in set:
            indx = string.find(char)
            if indx != -1 :
                break
    return indx

def repeat(string, ncopy):
    return string*ncopy

def verify(string, set, back=False):
    if len(set) == 0 :
        return -1
    if back :
        for char in string[-1]:
            indx = set.find(char)
            if indx == -1 :
                return string.rfind(char) + 1
    else:
        for char in string:
            indx = set.find(char)
            if indx == -1 :
                return string.find(char) + 1
    return 0

def maxloc(array, dim=None, mask=None):
    if isinstance(array, list):
        dummy = np.array(array)
    else:
        dummy = array
    if mask is not None:
        dummy = np.where(mask, dummy, dummy.min())
    if dim is not None :
        return np.argmax(dummy,axis=dim)
    else:
        if dummy.ndim > 1 :
            return np.unravel_index(np.argmax(dummy), dummy.shape)
        else:
            return np.argmax(dummy)

def minloc(array, dim=None, mask=None):
    if isinstance(array, list):
        dummy = np.array(array)
    else:
        dummy = array
    if mask is not None:
        dummy = np.where(mask, dummy, dummy.max())
    if dim is not None :
        return np.argmin(dummy,axis=dim)
    else:
        if dummy.ndim > 1 :
            return np.unravel_index(np.argmin(dummy), dummy.shape)
        else:
            return np.argmin(dummy)

def maxval(array, dim=None, mask=None):
    if isinstance(array, list):
        dummy = np.array(array)
    else:
        dummy = array
    if mask is not None:
        dummy = np.where(mask, dummy, dummy.min())
    if dim is not None :
        return np.amax(dummy,axis=dim)
    else:
        return np.amax(dummy)

def minval(array, dim=None, mask=None):
    if isinstance(array, list):
        dummy = np.array(array)
    else:
        dummy = array
    if mask is not None:
        dummy = np.where(mask, dummy, dummy.max())
    if dim is not None :
        return np.amin(dummy,axis=dim)
    else:
        return np.amin(dummy)

def pack(array, mask, vector=None):
    if vector is not None:
        _vector = np.copy(vector)
        dummy = array[np.where(mask)]
        num = len(dummy)
        _vector[:num] = dummy
        return _vector
    else:
        return array[np.where(mask)]

def test_FortranCompatible():
    list = [[1,3,5,7],
            [2,8,5,-1],
            [3,-3,3,0]]
    arr = np.array(list)
    print(">>> arr")
    print(arr)
    print(">>> findloc(arr, 5)")
    print(findloc(arr, 5))
    print(">>> count(arr < 0)")
    print(count(arr < 0))
    print(">>> pack(arr, arr > 2)")
    print(pack(arr, arr > 2))
    print(">>> pack(arr, arr > 2, [1,2,3,4,5,6,7,8,9])")
    print(pack(arr, arr > 2, [1,2,3,4,5,6,7,8,9]))

    print(">>> maxloc(arr)")
    print(maxloc(arr))
    print(">>> maxloc(arr, dim=1)")    
    print(maxloc(arr, dim=1))
    print(">>> maxloc(arr, mask=(arr < 6))")
    print(maxloc(arr, mask=(arr < 6)))

    print(">>> maxval(arr)")
    print(maxval(arr))
    print(">>> maxval(arr, dim=1)")    
    print(maxval(arr, dim=1))
    print(">>> maxval(arr, mask=(arr < 6))")
    print(maxval(arr, mask=(arr < 6)))

    print(">>> minloc(arr)")
    print(minloc(arr))
    print(">>> minloc(arr, dim=1)")    
    print(minloc(arr, dim=1))
    print(">>> minloc(arr, mask=(arr < 6))")
    print(minloc(arr, mask=(arr < 6)))

    print(">>> minval(arr)")
    print(minval(arr))
    print(">>> minval(arr, dim=1)")    
    print(minval(arr, dim=1))
    print(">>> minval(arr, mask=(arr < 6))")
    print(minval(arr, mask=(arr < 6)))

    str = "  This is a pen   "
    print(">>> str")
    print(str)
    print(">>> len(str)")
    print(len(str))
    print(">>> len_trim(str)")
    print(len_trim(str))
    print(">>> adjustl(str)")
    print(adjustl(str))
    print(">>> adjustr(str)")
    print(adjustr(str))
    print(">>> trim(str)")
    print(trim(str))
    print('>>> index(str, "is")')
    print(index(str, "is"))
    print('>>> scan(str, "lmnop")')
    print(scan(str, "lmnop"))
    print('>>> verify("fortran", "foo")')
    print(verify("fortran", "foo"))
    print(verify("fortran", "fortran"))
    print(verify("fortran", "c++", True))
    print(">>> repeat(str, 3)")  
    print(repeat(str, 3))   
    return
    

#------- index operation -------
def indgen(n,offset=0):
    return np.linspace(offset,n-1+offset,num=n,dtype = 'int')

def findgen(n,offset=0):
    return np.linspace(offset,n-1+offset,num=n)

def firstloc(mask, back=False):
    indx = np.where(mask)
    n = mask.ndim
    if back :
        i = -1
    else :
        i = 0
    if n == 1 :
        return indx[0][i]
    else :
        ans = []
        for j in range(n):
            ans.append(indx[i][j])
        return tuple(ans)

def indmask(mask):
    indx = np.where(mask)
    n = mask.ndim
    if n == 1 :
        return indx[0]
    else:
        return indx[:]

def replace(array, mask, replace):
    return np.where(mask, replace, array)

def test_IndexOperation():
    arr = findgen(20)
    print(">>> findgen(20)")
    print(arr)
    arr = arr.reshape(5,-1)
    print(">>> arr.reshape(5,-1)")
    print(arr)
    print(">>> indx0, indx1 = indmask(arr < 5.0)")
    indx0, indx1 = indmask(arr < 5.0)
    print(">> (indx0, indx1)")
    for i in range(len(indx0)):
        print("(",indx0[i],",",indx1[i],")")
    print(">>> replace(arr, arr < 5.0, 0.0)")
    print(replace(arr, arr < 5.0, 0.0))
    print(">>> firstloc(arr > 5.0)")
    pos = (firstloc(arr > 5.0))
    print(pos)
    label = 'arr[{},{}] = {}'.format(pos[0], pos[1], arr[pos])
    print(label)

#--------- String Operation ------------

def btrim(string):
    return string.strip()

def strzfill(num, len):
    return str(num).zfill(len)
    
def str2list(string):
    dummy = string.split('=')
    strs = dummy[1].strip(' ,').split(',')
    for i in range(len(strs)):
        strs[i] = strs[i].strip(' \'') 
    return strs    

#--------- Compatibility ------------

def checksep(dir, mode='add'):
    if dir[-1] == os.sep :
        if mode in ('del', 'DEL', 'Del', '-') : 
            return dir[:-1]
    else :
        if mode in ('add', 'ADD', 'Add', '+') :
            return dir + os.sep
    return dir

def pause(mssg='Press ENTER key to continue'):
    res = input(mssg)
    return

def delfile(filename):
    if os.path.exists(filename) :
        os.unlink(filename)

def dirscan(dir):
    path_list = glob.glob(dir + os.sep + '*')
    name = []
    ext = []
    for fullpath in path_list :
        file = os.path.basename(fullpath)
        _name, _ext = os.path.splitext(file)
        name.append(_name)
        ext.append(_ext)
    return name, ext

def seekdefault(filename, root=None):
    p = Path(filename)
    if p.is_file() :
        return filename
    else :
        home = Path.home()
        underhome = str(home) + os.sep + filename
        p = Path(underhome)
        if p.is_file() :
            return underhome
        else:
            if root is not None:
                special = checksep(root,'+') + filename
                p = Path(special)
                if p.is_file() :
                    return special
                else :
                    return None
            else :
                return None 

def strarr(shape, len=16):
    dtype = "|S{0:d}".format(len)
    return np.empty(shape, dtype)
    
def getLog():
    argvs = sys.argv 
    log = "{0} (UTC) Called in".format(strftime("%m/%d/%Y %H:%M", gmtime()))
    for val in argvs:
        log = log + " " + val
    return log

def readbin(binfile, buffsize, offset=0):
    import os
    try:
        if binfile.endswith('.gz'):
            import gzip
            import cStringIO 
            gid = gzip.open(binfile,'rb')
            gid.seek(offset*2, os.SEEK_SET)            
            str = cStringIO.StringIO()
            str = gid.read()
            raw = np.fromstring(str, np.int16, buffsize)
            gid.close()
        else:
            fid = open(binfile, 'rb')
            fid.seek(offset*2, os.SEEK_SET)
            raw = np.fromfile(fid, np.int16, buffsize)
            fid.close()
        return raw
    except Exception as ex:
         print("Error at readBin")
         print(ex.message)
         sys.exit(-1)   

def loadtxt(fname, dtype=float, comments='#', delimiter=',',
              converters=None, skiprows=0, usecols=None, unpack=False,
              ndmin=0, eoh=None):
    """
    Wrapper function of numpy.loadtxt
    Load data and list of lines commented out in header part of the text file.
    sys.stdin without eoh (end of header) is not allowed
    Each row in the text file must have the same number of values.
    Parameters
    ----------
    fname : file or str
        File, filename, or generator to read.  If the files with extensions of
        ``.gz`` and ``.bz2`` are noy suported.
    dtype : data-type, optional
        Data-type of the resulting array; default: float.  If this is a
        structured data-type, the resulting array will be 1-dimensional, and
        each row will be interpreted as an element of the array.  In this
        case, the number of columns used must match the number of fields in
        the data-type.
    comments : str or sequence, optional
        The characters or list of characters used to indicate the start of a
        comment;
        default: '#'.
    delimiter : str, optional
        The string used to separate values.  By default, this is any
        whitespace.
    converters : dict, optional
        A dictionary mapping column number to a function that will convert
        that column to a float.  E.g., if column 0 is a date string:
        ``converters = {0: datestr2num}``.  Converters can also be used to
        provide a default value for missing data (but see also `genfromtxt`):
        ``converters = {3: lambda s: float(s.strip() or 0)}``.  Default: None.
    skiprows : int, optional
        Skip the first `skiprows` lines; default: 0.
    usecols : sequence, optional
        Which columns to read, with 0 being the first.  For example,
        ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
        The default, None, results in all columns being read.
    unpack : bool, optional
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = loadtxt(...)``.  When used with a structured
        data-type, arrays are returned for each field.  Default is False.
    ndmin : int, optional
        The returned array will have at least `ndmin` dimensions.
        Otherwise mono-dimensional axes will be squeezed.
        Legal values: 0 (default), 1 or 2.
        .. versionadded:: 1.6.0
    eoh : str, optional
        The string used to detect the end of header.
    Returns
    -------
    out : ndarray
        Data read from the text file.
    header : 
        The list of lines commented out
    See Also
    --------
    numpy.loadtxt
    Notes
    -----
    This function aims to extend the `numpy.loadtxt`. The header section can be extracted.
    Examples
    --------
    >>> from io import StringIO   # StringIO behaves like a file object
    >>> c = StringIO("0 1\\n2 3")
    >>> np.loadtxt(c)
    array([[ 0.,  1.],
           [ 2.,  3.]])
    >>> d = StringIO("M 21 72\\nF 35 58")
    >>> np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),
    ...                      'formats': ('S1', 'i4', 'f4')})
    array([('M', 21, 72.0), ('F', 35, 58.0)],
          dtype=[('gender', '|S1'), ('age', '<i4'), ('weight', '<f4')])
    >>> c = StringIO("1,0,2\\n3,0,4")
    >>> x, y = np.loadtxt(c, delimiter=',', usecols=(0, 2), unpack=True)
    >>> x
    array([ 1.,  3.])
    >>> y
    array([ 2.,  4.])
    """
        
    def split_comment(line, regex_comments):
        """Chop off comments and strip.
        """
        buf = regex_comments.split(line, maxsplit=1)[0]
        buf = buf.strip('\r\n')
        if buf:
            return buf, ''
        else:
            line = regex_comments.split(line, maxsplit=1)[1]
            line = line.strip('\r\n')
            return None, line.lstrip()
    
    fown = False
    if isinstance(fname, str) :
        try :
            fd = open(fname, 'r')        
        except TypeError:
            raise ValueError('fname must be a filename')
        fown = True
    else :
        fd = fname
        
    for i in range(skiprows):
        dummy = fd.readline()
        
    header = []

    if comments is not None:
        if isinstance(comments, (str, bytes)):
            _comments = [comments]
        else:
            _comments = [comment for comment in comments]

        # Compile regex for comments beforehand
        _comments = (re.escape(comment) for comment in _comments)
        regex_comments = re.compile('|'.join(_comments))

        first_vals = None
        try:
            while True:
                pos = fd.tell()
                first_line = fd.readline()
                first_vals, commentout = split_comment(first_line, regex_comments)
                if first_vals :
                    break                   
                else :
                    header.append(commentout)
                if eoh :
                    if commentout.find(eoh) >= 0: break
        except StopIteration:
            # End of lines reached
            warnings.warn('readtxt: Empty input file: "%s"' % fname)
        if first_vals :
            try:
                fd.seek(pos)
            except TypeError:
                raise ValueError('sys.stdin without end_of_header not allowed')

    X = np.loadtxt(fd, dtype=dtype, comments=comments, delimiter=delimiter, converters=converters, skiprows=0,
                   usecols=usecols, unpack=unpack, ndmin=ndmin)
    
    if fown : fd.close()
    return X, header

def to_xny(x, ny):
    """
    Parameters
    ----------
    x: ndarray
        1 dimentional ndarray.  [x[0], x[1],...,x[m]]
    ny: ndarray
        2 dimentional array [[y1[0], y2[0], ..., yn[0]],
                            [y1[1], y2[1], ..., yn[1]],
                            ...
                            [y1[m], y2[m], ..., yn[m]]]
    Returns
    -------
    ndarray
        2 dimentional array [[x[0], y1[0], y2[0], ..., yn[0]],
                            [x[1], y1[1], y2[1], ..., yn[1]],
                            ...
                            [x[m], y1[m], y2[m], ..., yn[m]]]  

    See Also
    --------
    numpy.insert
    
    Notes
    -----

    """
    if ny.ndim == 1 :
        return np.vstack((x, ny)).transpose()
    else:
        return np.insert(ny, 0, x, axis=1)

def to_xyz(x, y, z):
    """
    Parameters
    ----------
    x: ndarray
        1 dimentional ndarray.  [x[0], x[1],...,x[m]] or
        2 dimentional ndarray.  [[x[0], x[0], ..., x[0]],
                               [x[1], x[1], ..., x[1]],
                                ...
                               [x[m], x[m], ..., x[m]]
                                   
    y: ndarray
        1 dimentional ndarray.  [y[0], y[1],...,y[n]] or
        2 dimentional array     [[y[0], y[1], ..., y[n]],
                                [y[0], y[1], ..., y[n]],
                                ...
                                [y[0], y[1], ..., y[n]]]
    z: ndarray
        2 dimentional array [[z[0,0], z[0,1], ..., z[0,n]],
                            [z[1,0], z[1,1], ..., z[1,n]],
                            ...
                            [z[m,0], z[m,1], ..., z[m,n]]]
    Returns
    -------
    ndarray
        2 dimentional array [[x[0], y[0], z[0,0]],
                            [x[0], y[1], z[0,1]],
                            ...
                            [x[0], y[n], z[0,n]],
                            [x[1], y[0], z[1,0]],
                            [x[1], y[1], z[1,1]],
                            ...
                            [x[1], y[n], z[1,n]],

    See Also
    --------
    numpy.insert
    
    Notes
    -----

    """
    m, n = z.shape
    if x.ndim == 1 :
        dummy = np.tile(x,(1,n)).reshape((n,m))
        dummy = dummy.transpose()
    else:
        dummy = x.transpose()
    xx = dummy.reshape(-1)
    if y.ndim == 1 :
        yy = np.tile(y,(1,m))
    else:
        yy = y.reshape(-1)
    zz = z.reshape(-1)
    return  np.vstack((xx, yy, zz)).transpose()
    
def savetxt(X, fname=None, fmt='%.18e', delimiter=' ', newline='\n', headers=None,
             footer='', comments='# ', append=False):
    """
    Wrapper function of numpy.savetxt

    Save an array to a text file.
    Parameters
    ----------
    X : array_like
        Data to be saved to a text file.
    fname : filename or file handle
        If the filename ends in ``.gz``, the file is automatically saved in
        compressed gzip format.  `loadtxt` understands gzipped files
        transparently.
    fmt : str or sequence of strs, optional
        A single format (%10.5f), a sequence of formats, or a
        multi-format string, e.g. 'Iteration %d -- %10.5f', in which
        case `delimiter` is ignored. For complex `X`, the legal options
        for `fmt` are:
            a) a single specifier, `fmt='%.4e'`, resulting in numbers formatted
                like `' (%s+%sj)' % (fmt, fmt)`
            b) a full string specifying every real and imaginary part, e.g.
                `' %.4e %+.4j %.4e %+.4j %.4e %+.4j'` for 3 columns
            c) a list of specifiers, one per column - in this case, the real
                and imaginary part must have separate specifiers,
                e.g. `['%.3e + %.3ej', '(%.15e%+.15ej)']` for 2 columns
    delimiter : str, optional
        String or character separating columns.
    newline : str, optional
        String or character separating lines.
        .. versionadded:: 1.5.0
    headers : str or sequence, optional
        String that will be written at the beginning of the file.
        .. versionadded:: 1.7.0
    footer : str, optional
        String that will be written at the end of the file.
        .. versionadded:: 1.7.0
    comments : str, optional
        String that will be prepended to the ``header`` and ``footer`` strings,
        to mark them as comments. Default: '# ',  as expected by e.g.
        ``numpy.loadtxt``.
        .. versionadded:: 1.7.0
    append : bool, optional
        If True, the array is appended to the file.
    See Also
    --------
    save : Save an array to a binary file in NumPy ``.npy`` format
    savez : Save several arrays into an uncompressed ``.npz`` archive
    savez_compressed : Save several arrays into a compressed ``.npz`` archive
    Notes
    -----
    Further explanation of the `fmt` parameter
    (``%[flag]width[.precision]specifier``):
    flags:
        ``-`` : left justify
        ``+`` : Forces to precede result with + or -.
        ``0`` : Left pad the number with zeros instead of space (see width).
    width:
        Minimum number of characters to be printed. The value is not truncated
        if it has more characters.
    precision:
        - For integer specifiers (eg. ``d,i,o,x``), the minimum number of
          digits.
        - For ``e, E`` and ``f`` specifiers, the number of digits to print
          after the decimal point.
        - For ``g`` and ``G``, the maximum number of significant digits.
        - For ``s``, the maximum number of characters.
    specifiers:
        ``c`` : character
        ``d`` or ``i`` : signed decimal integer
        ``e`` or ``E`` : scientific notation with ``e`` or ``E``.
        ``f`` : decimal floating point
        ``g,G`` : use the shorter of ``e,E`` or ``f``
        ``o`` : signed octal
        ``s`` : string of characters
        ``u`` : unsigned decimal integer
        ``x,X`` : unsigned hexadecimal integer
    This explanation of ``fmt`` is not complete, for an exhaustive
    specification see [1]_.
    References
    ----------
    .. [1] `Format Specification Mini-Language
           <http://docs.python.org/library/string.html#
           format-specification-mini-language>`_, Python Documentation.
    Examples
    --------
    >>> x = y = z = np.arange(0.0,5.0,1.0)
    >>> np.savetxt('test.out', x, delimiter=',')   # X is an array
    >>> np.savetxt('test.out', (x,y,z))   # x,y,z equal sized 1D arrays
    >>> np.savetxt('test.out', x, fmt='%1.4e')   # use exponential notation
    """
    def setHeaders(headers):
        _header = ""
        for line in headers:
            _header = _header + line + '\n'
        _header = _header[:-1]
        return _header

    try:
        if headers is not None:
            if isinstance(headers, list) :
                header = setHeaders(headers)
            else :
                header = setHeaders([headers]) 
        else:
            header = ''

#        print(header)
        if fname is None:
           if sys.version_info[0] >= 3:
               np.savetxt(sys.stdout.buffer, X, fmt, delimiter, newline, header, footer, comments)
           else:
               np.savetxt(sys.stdout, X, fmt, delimiter, newline, header, footer, comments)
        else:
           if append:
              fout = open(fname,"a")
              np.savetxt(fout, X, fmt, delimiter, newline, header, footer, comments)
              fout.close()
           else:
              np.savetxt(fname, X, fmt, delimiter, newline, header, footer, comments)
    except TypeError:
        raise ValueError('fname must be a string, file handle, or generator')              

def test_FileIO(fname):
    dat, header = loadtxt(fname, delimiter=',')
#    dat, header = loadtxt(fname, delimiter=',', eoh='[Data]')
    for i in range(len(header)):
        print(header[i])
    print(dat)
    print("----")
    n = dat.shape[0]
    vec = findgen(n)*0.2
    savetxt(to_xny(vec, dat[:,1:]), delimiter=', ', headers=header)
    print("=============")


def line2list(line):
    dummy = line.split('=')
    vals = dummy[1].split(',')
    for i in range(len(vals)):
        vals[i].strip(' \'') 
    return vals

def edf_formatted_date():
     return strftime("%m/%d/%Y %H:%M", localtime())

def listSearch(list, search_val) :
    index = []
    for i, e in enumerate(list):
        if search_val in e:
            index.append(i)
    return index
 
class edf():
    """
    edf() is pure python module for parsing a file written in experimantal data format (edf).
    if you want to open 'temp.edf', make instance of the file with edf(), 
    and then call load() method. 
    ex)
       exp = TE.edf()

    """
    def __init__(self):
        self.Name     = ''
        self.ShotNo   = 0
        self.SubNo    = 0
        self.Date     = ''
        self.DimNo    = 0
        self.DimSize  = []
        self.DimName  = []
        self.DimUnit  = []
        self.ValNo    = 0
        self.ValName  = []
        self.ValUnit  = []
        self.comments = []

    def clear(self):
        self.Name     = ''
        self.ShotNo   = 0
        self.SubNo    = 0
        self.Date     = ''
        self.DimNo    = 0
        self.DimSize  = []
        self.DimName  = []
        self.DimUnit  = []
        self.ValNo    = 0
        self.ValName  = []
        self.ValUnit  = []
        self.comments = []

    def show(self):
        print("Name =", self.Name)
        print("ShotNo =", self.ShotNo)
        print("SubNo =", self.SubNo)
        print("Date =", self.Date)
        print("DimNo =", self.DimNo)
        print("DimSize =", self.DimSize)
        print("DimName =", self.DimName)
        print("DimUnit =", self.DimUnit)
        print("ValNo =", self.ValNo)
        print("ValName =", self.ValName)
        print("ValUnit =", self.ValUnit)
        print("comments =", self.comments)
    
    def new(self, headers):
        """
        parsing a line of header.
        """
        reobjItem  = re.compile(r'(.+?)\s*=\s*(.+)')      # re.compile(r'(.+)\s*=\s*(.+)') 2025.10.14
        stat = False
        self.comments = []
        for line in headers:
            matchitem = reobjItem.match(line)
            if matchitem and not(stat):
                key = matchitem.groups()[0].upper()
                key = key.strip()
                val = matchitem.groups()[1]
                if 'NAME' == key:
                    clm = val.strip()
                    clm = val.strip('\'')
                    self.Name = clm
                if 'SHOTNO' == key:
                    self.ShotNo = int(val)
                if 'SUBNO' == key:
                    self.SubNo = int(val)
                if 'DATE' == key:
                    clm = val.strip()
                    clm = val.strip('\'')
                    self.Date = clm
                if 'DIMNO' == key:
                    self.DimNo = int(val)
                if 'DIMSIZE' == key:
                    clm = val.split(',')
                    for i in range(self.DimNo):
                        self.DimSize.append(int(clm[i]))
                if 'DIMNAME' == key:
                    clm = val.split(',')
                    for i in range(self.DimNo):
                        clmd = clm[i].strip()
                        clmd = clmd.strip('\'')
                        self.DimName.append(clmd)
                if 'DIMUNIT' == key:
                    clm = val.split(',')
                    for i in range(self.DimNo):
                        clmd = clm[i].strip()
                        clmd = clmd.strip('\'')
                        self.DimUnit.append(clmd)
                if 'VALNO' == key:
                    self.ValNo = int(val)
                if 'VALNAME' == key:
                    clm = val.split(',')
                    for i in range(self.ValNo):
                        clmd = clm[i].strip()
                        clmd = clmd.strip('\'')
                        self.ValName.append(clmd)
                if 'VALUNIT' == key:
                    clm = val.split(',')
                    for i in range(self.ValNo):
                        clmd = clm[i].strip()
                        clmd = clmd.strip('\'')
                        self.ValUnit.append(clmd)
            else:
                if line.upper().find("[DATA]") >= 0:
                    break
                if line.upper().find("[COMMENTS]") >= 0:
                    stat = True
                    continue
                if stat :
                    self.comments.append(line)      

                   
    def load(self, fname = None, usecols=None):
        if fname is not None :
            X, headers = loadtxt(fname, delimiter=',', usecols=usecols)
        else :             
            X, headers = loadtxt(sys.stdin, delimiter=',', usecols=usecols, eoh="[Data]")
        self.new(headers)
        if fname is not None:
            line = "original file is '{0}'".format(fname)
            self.comments.append(line)
        return X

    def save(self, X, fname=None, fmt='%.6e'):
        if fname is not None:
            line = "Written as '{0}'".format(fname)
            self.comments.append(line)
        self.comments.append(getLog())
        self.Date = strftime("%m/%d/%Y %H:%M", localtime())
        for i in range(self.DimNo):
             if i == 0:
                 dimname = "'{0}'".format(self.DimName[i])
                 dimsize = "{0:d}".format(self.DimSize[i])
                 dimunit = "'{0}'".format(self.DimUnit[i])
             else:
                 dimname = dimname + ", '{0}'".format(self.DimName[i])
                 dimsize = dimsize + ", {0:d}".format(self.DimSize[i])
                 dimunit = dimunit + ", '{0}'".format(self.DimUnit[i])

        for i in range(self.ValNo):
            if i == 0:
                valname = "'{0}'".format(self.ValName[i])
                valunit = "'{0}'".format(self.ValUnit[i])
            else:
                valname = valname + ", '{0}'".format(self.ValName[i])
                valunit = valunit + ", '{0}'".format(self.ValUnit[i])         

        _headers = []
        _headers.append("[Parameters]")
        _headers.append("Name = '{0}'".format(self.Name))
        _headers.append("ShotNo = {0:d}".format(self.ShotNo))
        _headers.append("SubNo = {0:d}".format(self.SubNo))
        _headers.append("Date = '{0}'".format(self.Date))
        _headers.append(" ")
        _headers.append("DimNo = {0:d}".format(self.DimNo))
        _headers.append("DimName = {0}".format(dimname))
        _headers.append("DimSize = {0}".format(dimsize))
        _headers.append("DimUnit = {0}".format(dimunit)) 
        _headers.append(" ")
        _headers.append("ValNo = {0:d}".format(self.ValNo))
        _headers.append("ValName = {0}".format(valname))
        _headers.append("ValUnit = {0}".format(valunit)) 
        _headers.append(" ")  
        _headers.append("[Comments]")
        for line in self.comments :
            _headers.append(line)
        _headers.append(" ")
        _headers.append("[Data]")
        if fname is not None:
            savetxt(X, fname=fname, fmt=fmt, delimiter=', ', headers=_headers, comments='# ')
        else :
            savetxt(X, fmt=fmt, delimiter=', ', headers=_headers, comments='# ')
        
    def setValNameFromNum(self, nums, fmt = '0.6E', unit = None):
        self.ValName = []
        self.ValUnit = []
        if unit is not None:
            unit = asbytes(unit)
        else:
            unit = ' '
        for val in nums :
            self.ValName.append("{"+fmt+"}".format(val))
            self.ValUnit.append(unit)

def xmlindent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            xmlindent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
    return

def default_parser(progname, description, version, timestamp):
    parser = argparse.ArgumentParser(
             prog = progname,
             description = description)
    parser.add_argument(
        '--version', 
        action='version', 
        version='Ver:{0} ({1})'.format(version, timestamp)
    )
    return parser
    
    
def setDefaultArguments(self, version, timestamp):
    self.add_argument(
        '--version', 
        action='version', 
        version='Ver:{0} ({1})'.format(version, timestamp))

def setFileArguments(self, default_wfmt='%18.e'):
    self.add_argument(
        '--ownformat', 
        action='store_true',
        default = False,
        help='If True, own format is used [ False ]')

    self.add_argument(
        '--usecols',
        action='store',
        nargs='+',
        type=int,
        default = None,
        metavar = 'int',
        help='Set columns to read, with 0 being the first [ All ]')
    
    self.add_argument(
        '--comment', 
        action='store',
        type=str,
        default = '#',
        metavar = 'str',
        help='A character used to indicate the start of a comment [ # ]')

    self.add_argument(
        '--delimiter', 
        action='store',
        type=str,
        default = ' ',
        metavar = 'str',
        help='Set delimiter used to separate values [ , ]')

    self.add_argument(
        '--skiprows', 
        action='store',
        type=int,
        default = 0,
        metavar = 'int',
        help='Skip the first "skiprows" lines [ 0 ]')
    
    self.add_argument(
        '--keepskipped', 
        action='store_true',
        default = False,
        help='If True, skipped lines are saved in headers [ False ]')    

    self.add_argument(
        '--wfmt', 
        action='store',
        type=str,
        default = default_wfmt,
        metavar = 'str',
        help='A single format, a sequence of formats, or a multi-format string [ %%.18E ]')   


def smart_split(string):
    """
    The string is split at locations where a comma (,) or whitespace character occurs one or more times.
    Consecutive commas or whitespace characters are treated as a single delimiter.
    A combination of whitespace and comma (e.g., , or ,) is also considered a single delimiter.
    Leading and trailing commas or whitespace characters are safely ignored.
    """
    return [p for p in re.split(r'[,\s]+', string.strip()) if p]

def list2float(lst):
    return [float(p) for p in lst]

def list2int(lst):
    return [int(p) for p in lst]

def select_elements(lst, indices):
    """
    Extract elements from a list at specified indices safely.

    Parameters
    ----------
    lst : list
        The source list.
    indices : list of int
        Indices of elements to extract.

    Returns
    -------
    list
        A new list containing elements at the valid indices.
        Invalid indices are ignored.
    """
    return [lst[i] for i in indices if 0 <= i < len(lst)]

def ensure_array(x):
    """Converts x to a NumPy array if it is a list."""
    if isinstance(x, list) :
        return np.array(x)
    else:
        return np.array([x])

if __name__== "__main__" :
    test_FortranCompatible()
    test_IndexOperation()
    test_FileIO("sample.dat")
