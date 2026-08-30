#! /usr/bin/env python
"""
    Plot utilities (matplotlib wrapper).

    Matplotlib Axes class and figure class should be inherited as below,
        matplotlib.axes.Axes.add_trace = MPU.add_trace
        matplotlib.axes.Axes.add_errorbar = MPU.add_errorbar
        matplotlib.axes.Axes.matview = MPU.matview
        matplotlib.figure.Figure.make_subplots = MPU.make_subplots

   function:
       mypage
       make_subplots
       add_trace
       add_errorbar
       matview       

    Dependence
    ----------

    Status
    ------
    Version 0.9.7

    Authour
    -------
    Shigeru Inagaki                                       
    Institute of Advanced Energy, Kyoto University
    inagaki.shigeru.7s@kyoto-u.ac.jp
    
    Revision History
    ----------------
    [20-Oct-2022] Creation                                  Ver 0.9.1
    [26-Oct-2022] Bug fixed                                 Ver 0.9.2
    [03-Oct-2023] mypage revised                            Ver 0.9.3
    [18-Sep-2025] mypage updated                            Ver 0.9.4
    [14-Oct-2025] Bug in addtrace/adderrobar fixed          Ver 0.9.5
    [16-Oct-2025] Bug in adderrobar fixed                   Ver 0.9.6
    [16-Oct-2025] legend_label_only added
                  ****kwargs added in adderrobar            Ver 0.9.7
    [13-Nov-2025] 'Liberation Sans' used instead of 
                  'Arial' in Linux                          Ver 0.9.8

    Copyright
    ---------
    2023 Shigeru Inagaki (inagaki.shigeru.7s@kyoto-u.ac.jp)
    Released under the CC BY-NC 4.0.

"""
import copy
import platform
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.ticker
import matplotlib.colors
import matplotlib.colorbar
import matplotlib.lines as mlines

_lines = ['-', '--', '-.', ':']
_markers = ["o","v","^","<",">","8","s","p","P","*","h","H","D","d","X"]

def mypage(backend=None, style='slide', portrait=False, scale=0.7, fontsize=None):
    os_name = platform.system()
    if backend is None:
        if os_name == 'Darwin':
             backend = 'Qt5Agg'

    if portrait :
        pagex = scale*8.27
        pagey = scale*11.69
        matplotlib.rcParams['savefig.orientation'] = 'portrait'
    else :
        pagex = scale*11.69
        pagey = scale*8.27
        matplotlib.rcParams['savefig.orientation'] = 'landscape'
        
    matplotlib.rcParams['ps.papersize'] = 'A4'
    
# classical style
    matplotlib.rcParams['lines.linewidth'] = 1.5
    matplotlib.rcParams['lines.markersize'] = 10
    matplotlib.rcParams['lines.markeredgewidth'] = 1.0
    matplotlib.rcParams['axes.linewidth'] = 1.2
    if backend is not None :
        matplotlib.rcParams['backend'] = backend    
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['xtick.labelsize'] = 'small'    
    matplotlib.rcParams['xtick.top'] = True
    matplotlib.rcParams['xtick.minor.visible'] = True
    matplotlib.rcParams['ytick.direction'] = 'in'
    matplotlib.rcParams['ytick.labelsize'] = 'small'   
    matplotlib.rcParams['ytick.right'] = True
    matplotlib.rcParams['ytick.minor.visible'] = True
    matplotlib.rcParams['ytick.minor.visible'] = True    

    matplotlib.rcParams['xtick.major.size'] = 6
    matplotlib.rcParams['xtick.minor.size'] = 3
    matplotlib.rcParams['xtick.major.width'] = 1.2
    matplotlib.rcParams['xtick.minor.width'] = 0.9  
    matplotlib.rcParams['ytick.major.size'] = 6
    matplotlib.rcParams['ytick.minor.size'] = 3
    matplotlib.rcParams['ytick.major.width'] = 1.2
    matplotlib.rcParams['ytick.minor.width'] = 1.0

    matplotlib.rcParams['figure.subplot.right'] = 0.95
    matplotlib.rcParams['figure.subplot.top'] = 0.9
    matplotlib.rcParams['figure.figsize'] = [pagex, pagey]  
        
# my favolite
    if os_name == 'Linux':
        matplotlib.rcParams['font.family'] = 'Liberation Sans'    #'sans-serif'
    else:
        matplotlib.rcParams['font.family'] = 'Arial'              #'sans-serif'
    if style == 'slide':        
        matplotlib.rcParams['figure.subplot.left'] = 0.15
        matplotlib.rcParams['figure.subplot.bottom'] = 0.12
        if fontsize is None :
            matplotlib.rcParams['font.size'] = 20.0
        else :
            matplotlib.rcParams['font.size'] = fontsize
        matplotlib.rcParams['legend.fontsize'] = 'x-small'
        matplotlib.rcParams['legend.title_fontsize'] = 'x-small'
    else:
        matplotlib.rcParams['figure.subplot.left'] = 0.12
        matplotlib.rcParams['figure.subplot.bottom'] = 0.095
        if fontsize is None :
            matplotlib.rcParams['font.size'] = 14.0
        else :
            matplotlib.rcParams['font.size'] = fontsize        
        matplotlib.rcParams['legend.fontsize'] = 'small'
        matplotlib.rcParams['legend.title_fontsize'] = 'small'

#    matplotlib.rcParams['savefig.orientation'] = 'landscape'
#    matplotlib.rcParams['ps.papersize'] = 'A4'
    fig = plt.figure()
    return fig

def AutoColor(n, reverse=True):
    if reverse :
        return  matplotlib.cm.jet_r(np.linspace(0, 1, n))
    else :
        return  matplotlib.cm.jet(np.linspace(0, 1, n))

def AutoLine(n, source=None):
    lines = []
    m = len(_lines)
    for i in range(n):
        if source is None :
            lines.append(_lines[i%m])
        else :
            lines.append(source)
    return lines

def AutoMarker(n, source=None):
    markers = []
    m = len(_markers)
    for i in range(n):
        if source is None :
            markers.append(_markers[i%m])
        else :
            markers.append(source)
    return markers

def make_subplots(self, nrows, ncols, nmax=None, shared_xaxes=True, roworder=True, left=0.15, bottom=0.15, right=0.95, top=0.9, wspace=0.2, hspace=0.15):
    """ return list of position of subplots 
        left  : 0.15
            The left side of the subplots on the figure 
   
        right : 0.95
            The right side of the subplots of the figure
    
        bottom : 0.15
            The bottom of the subplots of the figure
   
        top : 0.90
            The top of the subplots of the figure
   
        wspace : 
            The amount of width reserved for blank space between subplots, expressed as a fraction of the axis width
   
        hspace : 
            The amount of height reserved for white space between subplots, expressed as a fraction of the axis width
    """
    
    xlen = (right-left) / ((ncols-1)*wspace + ncols) 
    xgap = wspace*xlen
    ylen = (top-bottom) / ((nrows-1)*hspace + nrows) 
    ygap = hspace*ylen    

    if nmax is not None:
        _nmax = nmax
    else :
        _nmax = ncols*nrows

    if _nmax == 1 :
         ax = self.add_axes([left, top-ylen, xlen, ylen])
         return [ax]
    
    plots = []
    if roworder :    
        for j in range(ncols):
            for i in range(nrows):
                k = j*nrows + i
                if k > _nmax - 1:
                    break
                xorg = left + j*(xlen+xgap)
                yorg = top - ylen - i*(ylen+ygap)
                if shared_xaxes :
                    if k > 0:
                        ax = self.add_axes([xorg, yorg, xlen, ylen], sharex=plots[0])
                    else :
                        ax = self.add_axes([xorg, yorg, xlen, ylen])
                else :
                    ax = self.add_axes([xorg, yorg, xlen, ylen])
                plots.append(ax)
    else:
        for j in range(nrows):
            for i in range(ncols):
                k = j*ncols + i
                if k > _nmax :
                    break
                xorg = left + i*(xlen+xgap)
                yorg = top - ylen - j*(ylen+ygap)
                if shared_xaxes :
                    if k > 0 :
                        ax = self.add_axes([xorg, yorg, xlen, ylen], sharex=plots[0])
                    else :
                        ax = self.add_axes([xorg, yorg, xlen, ylen])
                else :
                    ax = self.add_axes([xorg, yorg, xlen, ylen])
                plots.append(ax)
    return plots

def to_scaler(val, default):
    if val is None :
        return default
    if isinstance(val,(list, tuple, np.ndarray)) :
        return val[0]
    else :
        return val

def add_trace(self, x, ys, xlabel='x', ylabel='y', mode='lines+markers', xlim=None, ylim=None, colors=None, ynames=None, markers=None, linestyles=None):  #, loc='best'):
    n = ys.ndim
    if n == 1 :
        if mode == 'lines+markers':
            self.plot(x, ys, color=to_scaler(colors,'b'), marker=to_scaler(markers,'o'), linestyle=to_scaler(linestyles,'-'))
        else :
            if mode == 'lines':
                self.plot(x, ys, color=to_scaler(colors,'b'), marker='', linestyle=to_scaler(linestyles,'-'))
            else :
                self.plot(x, ys, color=to_scaler(colors,'b'), marker=to_scaler(markers,'o'), linestyle='')
    else:
        (ny, nx) = ys.shape
        if markers is None :
            _markers_ = [_markers[0]]*ny
        else :
            if isinstance(markers,list) :
                _markers_ = markers
            else :
                _markers_ = [markers]*ny
                          
        if linestyles is None :
            _linestyles_ = [_lines[0]]*ny
        else :
            if isinstance(linestyles,list) :
                _linestyles_ = linestyles
            else :
                _linestyles_ = [linestyles]*ny
                          
        if colors is None :
            _colors_ = AutoColor(ny)
        else :
            if isinstance(colors,(list,np.ndarray)) :
                _colors_ = colors
            else :
                _colors_ = [colors]*ny
                          
        if ynames is None :
            labels = []
            for i in range(ny):
                labels.append('y{}'.format(i+1))
        else :
            labels = ynames
                          
        if mode == 'lines+markers':                          
            for i in range(ny):
                self.plot(x, ys[i,:], color=_colors_[i], label=labels[i], marker=_markers_[i], linestyle=_linestyles_[i])
        else :
            if mode == 'lines':
                for i in range(ny):
                    self.plot(x, ys[i,:], color=_colors_[i], label=labels[i], marker='', linestyle=_linestyles_[i])
            else :
                for i in range(ny):
                    self.plot(x, ys[i,:], color=_colors_[i], label=labels[i], marker=_markers_[i], linestyle='')
#        if ny > 1 :
#            self.legend(loc=loc)
                          
    if xlim is not None:
        self.set_xlim(xlim[0], xlim[1])
        if ylim is None :
            indx = np.where(np.logical_and(x >= xlim[0], x <= xlim[1]))
            if n > 1 :
                ymax = ys[:,indx[0]].max()
                ymin = ys[:,indx[0]].min()
            else :
                ymax = ys[indx[0]].max()
                ymin = ys[indx[0]].min()
            self.set_ylim(ymin, ymax)
    if ylim is not None:
        self.set_ylim(ylim[0], ylim[1])
    self.set_xlabel(xlabel)
    self.set_ylabel(ylabel)
    return

def add_errorbar(self, x, ys, xerr=None, yerr=None, xlabel='x', ylabel='y', mode='lines+markers', xlim=None, ylim=None, colors=None, ynames=None, markers=None, linestyles=None,**kwargs):   #, loc='best'):
    n = ys.ndim
    if n == 1 :
        if mode == 'lines+markers':
            if xerr is not None :
                if yerr is not None :                          
                    self.errorbar(x, ys, xerr=xerr, yerr=yerr, color=to_scaler(colors,'b'), fmt=to_scaler(markers,'o'), linestyle=to_scaler(linestyles,'-'),**kwargs)
                else :
                    self.errorbar(x, ys, xerr=xerr, color=to_scaler(colors,'b'), fmt=to_scaler(markers,'o'), linestyle=to_scaler(linestyles,'-'),**kwargs)
            else:
                if yerr is not None :                          
                    self.errorbar(x, ys, yerr=yerr, color=to_scaler(colors,'b'), fmt=to_scaler(markers,'o'), linestyle=to_scaler(linestyles,'-'),**kwargs)
                else :
                    self.plot(x, ys, color=to_scaler(colors,'b'), marker=to_scaler(markers,'o'), linestyle=to_scaler(linestyles,'-'),**kwargs)                                  
        else :
            if mode == 'lines':
                if xerr is not None :
                    if yerr is not None :            
                        self.errorbar(x, ys, xerr=xerr, yerr=yerr, color=to_scaler(colors,'b'), fmt='', linestyle=to_scaler(linestyles,'-'),**kwargs)
                    else :
                        self.errorbar(x, ys, xerr=xerr, color=to_scaler(colors,'b'), fmt='', linestyle=to_scaler(linestyles,'-'),**kwargs)
                else :
                    if yerr is not None :
                        self.errorbar(x, ys, yerr=yerr, color=to_scaler(colors,'b'), fmt='', linestyle=to_scaler(linestyles,'-'),**kwargs)
                    else :
                        self.plot(x, ys, color=to_scaler(colors,'b'), marker='', linestyle=to_scaler(linestyles,'-'),**kwargs)
            else :
                if xerr is not None :
                    if yerr is not None : 
                        self.errorbar(x, ys, xerr=xerr, yerr=yerr, color=to_scaler(colors,'b'), fmt=to_scaler(markers,'o'), linestyle='',**kwargs)
                    else:
                        self.errorbar(x, ys, xerr=xerr, color=to_scaler(colors,'b'), fmt=to_scaler(markers,'o'), linestyle='',**kwargs)
                else :
                    if yerr is not None : 
                        self.errorbar(x, ys, yerr=yerr, color=to_scaler(colors,'b'), fmt=to_scaler(markers,'o'), linestyle='',**kwargs)
                    else :
                        self.plot(x, ys, color=to_scaler(colors,'b'), marker=to_scaler(markers,'o'), linestyle='',**kwargs)
    else:
        (ny, nx) = ys.shape
        if markers is None :
            _markers_ = [_markers[0]]*ny
        else :
            if isinstance(markers,list) :
                _markers_ = markers
            else :
                _markers_ = [markers]*ny
                          
        if linestyles is None :
            _linestyles_ = [_lines[0]]*ny
        else :
            if isinstance(linestyles,list) :
                _linestyles_ = linestyles
            else :
                _linestyles_ = [linestyles]*ny
                          
        if colors is None :
            _colors_ = AutoColor(ny)
        else :
            if isinstance(colors,(list,np.ndarray)) :
                _colors_ = colors
            else :
                _colors_ = [colors]*ny
                          
        if ynames is None :
            labels = []
            for i in range(ny):
                labels.append('y{}'.format(i+1))
        else :
            labels = ynames
                          
        if mode == 'lines+markers':
            if xerr is not None :
                if yerr is not None : 
                    for i in range(ny):
                        self.errorbar(x, ys[i,:], xerr=xerr, yerr=yerr[i,:], color=_colors_[i], label=labels[i], fmt=_markers_[i], linestyle=_linestyles_[i],**kwargs)
                else :
                    for i in range(ny):
                        self.errorbar(x, ys[i,:], xerr=xerr, color=_colors_[i], label=labels[i], fmt=_markers_[i], linestyle=_linestyles_[i],**kwargs)                                  
            else :
                if yerr is not None :
                    for i in range(ny):
                        self.errorbar(x, ys[i,:], yerr=yerr[i,:], color=_colors_[i], label=labels[i], fmt=_markers_[i], linestyle=_linestyles_[i],**kwargs)
                else :
                    for i in range(ny):
                        self.plot(x, ys[i,:], color=_colors_[i], label=labels[i], marker=_markers_[i], linestyle=_linestyles_[i],**kwargs)
        else :
            if mode == 'lines':
                if xerr is not None :
                    if yerr is not None :
                        for i in range(ny):
                            self.errorbar(x, ys[i,:], xerr=xerr, yerr=yerr[i,:], color=_colors_[i], label=labels[i], fmt='', linestyle=_linestyles_[i],**kwargs)
                    else :
                        for i in range(ny):
                            self.errorbar(x, ys[i,:], xerr=xerr, color=_colors_[i], label=labels[i], fmt='', linestyle=_linestyles_[i],**kwargs)                                  
                else :
                    if yerr is not None : 
                        for i in range(ny):
                            self.errorbar(x, ys[i,:], yerr=yerr[i,:], color=_colors_[i], label=labels[i], fmt='', linestyle=_linestyles_[i],**kwargs)
                    else :
                        for i in range(ny):
                            self.plot(x, ys[i,:], color=_colors_[i], label=labels[i], marker='', linestyle=_linestyles_[i],**kwargs)          
            else :
                if xerr is not None :
                    if yerr is not None :
                        for i in range(ny):
                            self.errorbar(x, ys[i,:], xerr=xerr, yerr=yerr[i,:], color=_colors_[i], label=labels[i], fmt=_markers_[i], linestyle='',**kwargs)
                    else :
                        for i in range(ny):
                            self.errorbar(x, ys[i,:], xerr=xerr, color=_colors_[i], label=labels[i], fmt=_markers_[i], linestyle='',**kwargs)
                else :
                    if yerr is not None :
                        for i in range(ny):
                            self.errorbar(x, ys[i,:], yerr=yerr[i,:], color=_colors_[i], label=labels[i], fmt=_markers_[i], linestyle='',**kwargs)
                    else :
                        for i in range(ny):
                            self.plot(x, ys[i,:], color=_colors_[i], label=labels[i], marker=_markers_[i], linestyle='',**kwargs)
#        if ny > 1 :
#            self.legend(loc=loc)
                          
    if xlim is not None:
        self.set_xlim(xlim[0], xlim[1])
        if ylim is None :
            indx = np.where(np.logical_and(x >= xlim[0], x <= xlim[1]))
            if n > 1 :
                ymax = ys[:,indx[0]].max()
                ymin = ys[:,indx[0]].min()
            else :
                ymax = ys[indx[0]].max()
                ymin = ys[indx[0]].min()
            self.set_ylim(ymin, ymax)
    if ylim is not None:
        self.set_ylim(ylim[0], ylim[1])
    self.set_xlabel(xlabel)
    self.set_ylabel(ylabel)
    return

def matview(self, x, y, z, cmap=None, nbins=15, xaxis=-1, zmin=None, zmax=None, zscale_log=False):
    xmesh = 1.5*x[:-1]-0.5*x[1:]
    xmesh = np.append(xmesh, 0.5*(x[-1]+x[-2]))
    ymesh = 1.5*y[:-1]-0.5*y[1:]
    ymesh = np.append(ymesh, 0.5*(y[-1]+y[-2]))    
    if zmin is None :
        zmin = z.min()
    if zmax is None :
        zmax = z.max()
    if zscale_log :
        zmin = np.log10(zmin)
        zmax = np.log10(zmax)
    levels = matplotlib.ticker.MaxNLocator(nbins=nbins).tick_values(zmin, zmax)
    if cmap is None:
        _cmap =  matplotlib.cm.get_cmap('jet')
    else:
        _cmap =  matplotlib.cm.get_cmap(cmap)
    nm =  matplotlib.colors.BoundaryNorm(levels, ncolors=_cmap.N, clip=True)

    if xaxis == 0 :
        z = z.transpose()
    if zscale_log :
        image = self.pcolormesh(xmesh, ymesh, np.log10(z), cmap=_cmap, norm=nm)
    else:
        image = self.pcolormesh(xmesh, ymesh, z, cmap=_cmap, norm=nm)
    if xaxis == 0 :
        z = z.transpose()    
    plt.colorbar(image)   
      
def xyzFromeg1D(x, y, z_x):
   xx, yy = np.meshgrid(x,y)
   zz = z_x.transpose()
   return xx, yy, zz

def legend_label_only(label):
    return mlines.Line2D([], [], color='none', label=label)
