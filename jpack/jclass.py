#!/usr/bin/env python
# coding: utf-8
"""
    This module provides common classes and functions used across the jpack application.

    jview
        TraceBase
        FrameBase
        NTRACE_TAB
        NFRAME_TAB
        NTAB_TRACE
        NTAB_FRAME
        loadfromjson
        saveasjson

    jgraph
        split2int
        loadfromjson

    edfview
        edfview_1D

    Dependence
    ----------
    turnelib.py
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
    [19-Sep-2025] Creation                                  Ver 1.0
        
    Copyright
    ---------
    2025 Shigeru Inagaki (inagaki.shigeru.7s@kyoto-u.ac.jp)
    Released under the CC BY-NC 4.0.

"""
import json
import turnelib as TE
import matplotlib
import matplotlib.pyplot as plt
import matplotlibutil as MPU

matplotlib.axes.Axes.add_trace = MPU.add_trace
matplotlib.figure.Figure.make_subplots = MPU.make_subplots

NTRACE_TAB = 10
NFRAME_TAB = 10
NTAB_TRACE = 6
NTAB_FRAME = 2
# ntracemax <= NTRACE_TAB * NTAB_TRACE
# nframemax <= NFRAME_TAB * NTAB_FRAME

class TraceBase():
    def __init__(self, obj=None):
        if obj is None :
            self.id = 0
            self.visible = True
            self.frame = 1
            self.name = ''
            self.strshot = ''
            self.vals = ''  # all, '1 3 5 7 8'
            self.tags = ''  # use_ValName, use_shotnum, use_name, 'ch1 ch2 ch3 ch4'
            self.nskip = 1
            self.scale = 1.0
            self.offset = 0.0
            self.xmask = False
            self.xmin = 0.0
            self.xmax = 5.0
            self.ymask = False
            self.ymin = 0.0
            self.ymax = 5.0
            self.color = 'auto'
        else:
            self.id = obj['id']
            self.visible = obj['visible']
            self.frame = obj['frame']
            self.name = obj['name']
            self.strshot = obj['strshot']
            self.vals = obj['vals']
            self.tags = obj['tags']
            self.nskip = obj['nskip']
            self.scale = obj['scale']
            self.offset = obj['offset']
            self.xmask = obj['xmask']
            self.xmin = obj['xmin']
            self.xmax = obj['xmax']
            self.ymask = obj['ymask']
            self.ymin = obj['ymin']
            self.ymax = obj['ymax']
            self.color = obj['color']
        return
    
    def show(self):
        print(self.id)
        print(self.visible)
        print(self.frame)
        print(self.name)
        print(self.strshot)
        print(self.vals)
        print(self.tags)
        print(self.nskip)
        print(self.scale)
        print(self.offset)
        print(self.xmask)
        print(self.xmin)
        print(self.xmax)
        print(self.ymask)
        print(self.ymin)
        print(self.ymax)
        print(self.color) 
        return

    def update(self, values):
        id = self.id
        key2 = '-visible{}-'.format(id)
        key3 = '-frame{}-'.format(id)            
        key4 = '-name{}-'.format(id)   
        key5 = '-strshot{}-'.format(id)   
        key6 = '-vals{}-'.format(id)              
        key7 = '-tags{}-'.format(id)      
        key8 = '-nskip{}-'.format(id)   
        key9 = '-scale{}-'.format(id)  
        key10 = '-offset{}-'.format(id)     
        key11 = '-xmask{}-'.format(id)       
        key12 = '-xmaskmin{}-'.format(id)   
        key13 = '-xmaskmax{}-'.format(id)   
        key14 = '-ymask{}-'.format(id)   
        key15 = '-ymaskmin{}-'.format(id)   
        key16 = '-ymaskmax{}-'.format(id) 
        key17 = '-color{}-'.format(id)

        self.visible = values[key2]
        self.frame = int(values[key3])
        self.name = values[key4]
        self.strshot = values[key5]
        self.vals = values[key6]
        self.tags = values[key7]
        self.nskip = int(values[key8])
        self.scale = float(values[key9])
        self.offset = float(values[key10])
        self.xmask = values[key11]
        self.xmin = float(values[key12])
        self.xmax = float(values[key13])
        self.ymask = values[key14]
        self.ymin = float(values[key15])
        self.ymax = float(values[key16])
        self.color = values[key17]
        return

    def todict(self):
        d = dict(
            id = self.id,
            visible = self.visible,
            frame = self.frame,
            name = self.name,
            strshot = self.strshot,
            vals = self.vals,
            tags = self.tags,
            nskip = self.nskip,
            scale = self.scale,
            offset = self.offset,
            color = self.color,
            xmask = self.xmask,
            xmin = self.xmin,
            xmax = self.xmax,
            ymask = self.ymask,
            ymin = self.ymin,
            ymax = self.ymax
            )
        return d

class FrameBase():
    def __init__(self, obj=None):
        if obj is None:
            self.id = 0
            self.frameid = 0
            self.draw = True
            self.xname = ''
            self.xrangefix = False
            self.xmin = 0.0
            self.xmax = 5.0
            self.yname = ''
            self.yrangefix = False
            self.ymin = 0.0
            self.ymax = 5.0
            self.legend = True
        else:
#            self.id = obj['id']
            self.frameid = obj['frameid']
            self.id = obj.get('id',self.frameid)
            self.draw = obj['draw']
            self.xname = obj['xname']
            self.xrangefix = obj['xrangefix']
            self.xmin = obj['xmin']
            self.xmax = obj['xmax']
            self.yname = obj['yname']
            self.yrangefix = obj['yrangefix']
            self.ymin = obj['ymin']
            self.ymax = obj['ymax']
            self.legend = obj.get('legend', True)
        return
        
    def update(self, values):
        id = self.id
        key1 = '-fid{}-'.format(id)
        key2 = '-draw{}-'.format(id)   
        key3 = '-xname{}-'.format(id)   
        key4 = '-xrangefix{}-'.format(id)       
        key5 = '-xrangemin{}-'.format(id)   
        key6 = '-xrangemax{}-'.format(id)   
        key7 = '-yname{}-'.format(id)   
        key8 = '-yrangefix{}-'.format(id)       
        key9 = '-yrangemin{}-'.format(id)   
        key10 = '-yrangemax{}-'.format(id)
        self.frameid = int(values[key1])
        self.draw = values[key2]
        self.xname = values[key3]
        self.xrangefix = values[key4]
        self.xmin = float(values[key5])
        self.xmax = float(values[key6])
        self.yname = values[key7]
        self.yrangefix = values[key8]
        self.ymin = float(values[key9])
        self.ymax = float(values[key10])
        return

    def todict(self):
        d = dict(
#            id = self.id,
            frameid = self.frameid,
            draw = self.draw,
            xname = self.xname,
            xrangefix = self.xrangefix,
            xmin = self.xmin,
            xmax = self.xmax,
            yname = self.yname,
            yrangefix = self.yrangefix,
            ymin = self.ymin,
            ymax = self.ymax,
            legend = self.legend
            )
        return d

class ViewBase():
    def __init__(self, params=None):
        if params is None :
            self.device = 'Heliotron J'
            self.jview = True
            self.datapath = ''
            self.multishot = False
            self.shotnum = 12345
            self.plotstyle = 'TEVO'   # TEVO or RPRO
            self.xunit = 'ms'         # s ms or m cm
            self.dprf = 'hj'
            self.fext = 'edf'
            self.ncol = 1
            self.nrow = 1
            self.pagestyle = 'slide'        # slide or paper
            self.orientation = 'landscape'  # landscape or portrait
            self.automargin = True
            self.left = 0.1
            self.right = 0.1
            self.top = 0.1
            self.bottom = 0.1
            self.autospace = True
            self.wspace = 0.2
            self.hspace = 0.1
            self.fontsize = 'auto'
            self.title = '{device} {shot} {condition}'
            self.autotitle = True
            self.traces = []
            for i in range(NTRACE_TAB*NTAB_TRACE):
                t = TraceBase()
                t.id = i + 1   
                t.name = '--'
                self.traces.append(t)
            self.frames = []
        else :
            self.device = params[0]['device']
            self.jview =  params[0].get('jview', True)                   # params[0]['jview']
            self.datapath = params[0]['datapath']
            self.multishot = params[0]['multishot']
            self.shotnum = params[0]['shotnum']
            self.plotstyle = params[0]['plotstyle']
            self.xunit = params[0]['xunit']
            self.dprf = params[0]['dprf']
            self.fext = params[0]['fext']
            self.ncol = params[0]['ncol']
            self.nrow = params[0]['nrow']
            self.pagestyle = params[0]['pagestyle']
            self.orientation = params[0].get('orientation', 'landscape') # params[0]['orientation']
            self.automargin = params[0].get('automargin',True)           # params[0]['automargin']
            self.left = params[0].get('left',0.1)                        # params[0]['left']
            self.right = params[0].get('right',0.1)                      # params[0]['right']
            self.top = params[0].get('top',0.1)                          # params[0]['top']
            self.bottom = params[0].get('bottom',0.1)                    # params[0]['bottom']
            self.autospace = params[0].get('autospace',True)             # params[0]['autospace']
            self.wspace = params[0].get('wspace',0.2)                    # params[0]['wspace']
            self.hspace = params[0].get('hspace',0.1)                    # params[0]['hspace'] 
            self.fontsize = params[0]['fontsize']
            self.title = params[0]['title']
            self.autotitle = params[0]['autotitle']
            self.traces = []
            for obj in params[1]:
                t = TraceBase(obj)
                self.traces.append(t)  
            self.frames = []
            for obj in params[2]:
                f = FrameBase(obj)
                self.frames.append(f)  
        return                 

    def show(self):
        print(self.device)
        print(self.jview)
        print(self.datapath)
        print(self.multishot)
        print(self.shotnum)
        print(self.plotstyle)
        print(self.xunit)
        print(self.dprf)
        print(self.fext)
        print(self.ncol)
        print(self.nrow)
        print(self.pagestyle)
        print(self.orientation)
        print(self.automargin)
        print(self.left)
        print(self.right)
        print(self.top)
        print(self.bottom)
        print(self.autospace)
        print(self.wspace)
        print(self.hspace)
        print(self.fontsize)
        print(self.title)
        print(self.autotitle)  
        for t in self.traces:
            t.show()
        for f in self.frames:
            f.show()
        return 
    
    def update(self, values):
        self.datapath = TE.checksep(values['-datapath-'])
        self.multishot = values['-multishot-']
        self.shotnum = int(values['-shotnum-'])
        self.plotstyle = values['-plotstyle-']
        self.xunit = values['-xunit-']
        self.dprf = values['-dprf-']
        self.fext = values['-fext-']
        self.ncol = int(values['-cols-'])
        self.nrow = int(values['-rows-'])
        self.pagestyle = values['-pagestyle-']
        self.fontsize = values['-fontsize-']
        self.title = values['-title-']
        self.autotitle = values['-autotitle-']
        for f in self.frames:
            f.update(values)
        for t in self.traces:
            t.update(values)                
        return       

    def toobj(self):
        obj = []
        d1 = dict(
            __comment__ = "setup",
            device = self.device,
            jview = self.jview,
            datapath = self.datapath,
            multishot = self.multishot,
            shotnum = self.shotnum,
            plotstyle = self.plotstyle,
            xunit = self.xunit,
            dprf = self.dprf,
            fext = self.fext,
            ncol = self.ncol,
            nrow = self.nrow,
            pagestyle = self.pagestyle,
            orientation = self.orientation,
            automargin = self.automargin,
            left = self.left,
            right = self.right,
            top = self.top,
            bottom = self.bottom,
            autospace = self.autospace,
            wspace = self.wspace,
            hspace = self.hspace,
            fontsize = self.fontsize,
            title = self.title,
            autotitle = self.autotitle
            )
        obj.append(d1)
        traces = []
        for t in self.traces:
            if t.visible :
                traces.append(t.todict())
        obj.append(traces)
        frames = []
        for f in self.frames:
            if f.draw:
                frames.append(f.todict())
        obj.append(frames)
        return obj 

def loadfromjson(filename):
    with open(filename, 'r') as f:
        obj = json.load(f)
    view = ViewBase(params=obj)
    ntrace = len(view.traces)
    ntracemax = NTRACE_TAB*NTAB_TRACE
    if ntrace < ntracemax:
        for id in range(ntrace+1,ntracemax+1):
            t = TraceBase()
            t.id = id
            t.visible = False
            view.traces.append(t)
    nframe = len(view.frames)
    nframemax = NFRAME_TAB*NTAB_FRAME
    if nframe < nframemax:
        for id in range(nframe+1,nframemax+1):
            f = FrameBase()
            f.id = id
            f.frameid = id
            f.draw = False
            view.frames.append(f)
    return view
            
def saveasjson(view, filename):
    params = view.toobj()
    with open(filename, 'w') as f:
        json.dump(params,f)
    return

def int_up(nume, deno):
    return -(-nume//deno)    

def split2int(line):
    dummy = line.strip().split()
    res = []
    for item in dummy:
        res.append(int(item))
    return res

def split2float(line):
    dummy = line.strip().split()
    res = []
    for item in dummy:
        res.append(float(item))
    return res

#def list_copy_index(source, index):
#    output = []
#    for i in index:
#        output.append(source[i])
#    return output

def edfview_1D(x, ys, xlabel, ylabel, ynames, title, id_plot = None, multi=False, portrait=False, reverse=True):
    """ys[nplot,xlen]"""
    nplot = len(ys[:,0])  
    if id_plot is not None:
        ys = ys[id_plot,:]
#        ynames = list_copy_index(ynames, id_plot)
        ynames = TE. select_elements(ynames, id_plot)
        nplot = len(id_plot)

    _colors = MPU.AutoColor(nplot, reverse=reverse)
    
    if multi :
        if nplot <= 10 :
            nrow = nplot
            ncol = 1
        else :
            if nplot <= 20 :
                if portrait:
                    nrow = nplot
                    ncol = 1
                else:
                    ncol = 2
                    nrow = (nplot + 1)//2
            else :
                if nplot <= 30 :
                    if portrait:
                        ncol = 2
                        nrow = (nplot + 1)//2
                    else:
                        ncol = 3
                        nrow = (nplot + 2)//3
                else :
                    if portrait:
                        nrow = 20
                        ncol = (nplot + 19)//20
                    else:
                        nrow = 10
                        ncol = (nplot + 9)//10
        nres = ncol*nrow - nplot 
    else:
        nrow = 1
        ncol = 1

    if nrow > 6 :
        page = MPU.mypage(fontsize=14, scale=0.84, style='slide', portrait=portrait)
    else :
        page = MPU.mypage(fontsize=20, scale=0.84, style='slide', portrait=portrait)
    plots = page.make_subplots(nrow,ncol)

    if nrow == 1 :
        plots[0].add_trace(x, ys, xlabel=xlabel, ylabel=ylabel, ynames=ynames, mode='lines', colors=_colors)
        plots[0].legend(ynames,loc='best',frameon=False)
        plots[0].set_title(title, loc='right', fontdict={'fontfamily':'monospace','fontsize':12,})
    else :
        k = 0
        for j in range(ncol) :
            for i in range(nrow) :
                if k > nplot -1 :
                    break
                if i < nrow - 1 and k < nplot - 1:
                    if ylabel :
                        plots[k].add_trace(x, ys[k,:], xlabel='', ylabel='', mode='lines', colors=[_colors[k]])
                        plots[k].legend([ynames[k]],loc='best',frameon=False,handlelength=0,handletextpad=0.5)
                    else :
                        plots[k].add_trace(x, ys[k,:], xlabel='', ylabel=ynames[k], mode='lines', colors=[_colors[k]])  
                    plt.setp(plots[k].get_xticklabels(), visible=False)  
                else :
                    if ylabel :
                        plots[k].add_trace(x, ys[k,:], xlabel=xlabel, ylabel='', mode='lines', colors=[_colors[k]])
                        plots[k].legend([ynames[k]],loc='best',frameon=False,handlelength=0,handletextpad=0.5)
                    else :
                        plots[k].add_trace(x, ys[k,:], xlabel=xlabel, ylabel=ynames[k], mode='lines', colors=[_colors[k]])                        
                k = k + 1
        if nres > 0 :
            for j in range(nres):
                plots[j+nplot].axis('off')
        page.suptitle(title, fontsize=12, x=0.95, ha='right', y=0.92)
        if multi :
            page.supylabel(ylabel, x=0.05, fontsize=18)         
    plt.show()
    return
