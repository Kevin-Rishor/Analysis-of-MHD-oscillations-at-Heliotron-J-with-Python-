#!/usr/bin/python
# coding: utf-8
"""
    Device-dependent functions

    Dependence
    ----------
    turnelib

    Status
    ------
    Version 1.1

    Authour
    -------
    Shigeru Inagaki                                       
    Institute of Advanced Energy, Kyoto University
    inagaki.shigeru.7s@kyoto-u.ac.jp  
    
    Revision History
    ----------------
    [06-Apr-2025] Creation                               Ver 1.0
    [19-Oct-2025] igetfile added                       Ver 1.1  
                            
    Copyright
    ---------
    2025 Shigeru Inagaki (inagaki.shigeru.7s@kyoto-u.ac.jp)
    Released under the CC BY-NC 4.0.

"""
import io
import string
import urllib3
import traceback
import turnelib as TE

VERSION = '1.1'

BASEURL="https://exp.lhd.nifs.ac.jp/opendata/LHD/webapi.fcgi?cmd=getfile&diag=%s&shotno=%d&subno=%d"

def getdiagnames(dirname,device=None):
    """
    Heliotron J only
    """
    diagnames = []
    if device is None or device == "Heliotron J" :
        name = []
        ext = []
        _name, _ext = TE.dirscan(dirname)
        name.extend(_name)
        ext.extend(_ext)
        for i in range(len(name)):
            if ext[i] == '.edf' :
                dummy = name[i].split('@')
                if len(dummy) > 1:
                    diagnames.append(dummy[0])
        diagnames.sort()
    return diagnames

def igetfile(self, diag, shotno, subno=1):
    """
    LHD only
    read edf data from open data server.
    """
    url=BASEURL % (diag, shotno, subno)
    http=urllib3.PoolManager()
    dat = None
    try:
        resp = http.request('GET', url)
        fileio=io.StringIO(resp.data.decode('utf-8')) 
        dat = self.load(fileio)
    except:
        print(traceback.format_exc())
        pass
     
    return dat



