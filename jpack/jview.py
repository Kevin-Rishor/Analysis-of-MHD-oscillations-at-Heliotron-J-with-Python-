#!/usr/bin/env python
# coding: utf-8
"""
    This application makes a setup file for jgraph.py, which
    visualizes edf-formatted files.

    Usage:
    hjparam [-h] [--version] [-j filename] 

    optional arguments:
      -h, --help              show this help message and exit
      --version              show program's version number and exit
      -j filename, --json filename
                                  json-file name for initial setting

    Example:
      jview.py -j myjview.json

    Dependence
    ----------
    jslang.py
    jclass.py
    turnelib.py
    hjgraph.py
    TkEasyGUI

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
    [06-Apr-2025] Creation                                  Ver 1.0
    [19-Sep-2025] ViewBase has been updated for local mode  Ver 1.1
    [19-Sep-2025] The classes have been separated into an
                  independent module jclass                 Ver 1.2
        
    Copyright
    ---------
    2025 Shigeru Inagaki (inagaki.shigeru.7s@kyoto-u.ac.jp)
    Released under the CC BY-NC 4.0.

"""
import argparse
import subprocess
import TkEasyGUI as eg
import jslang as JS
import json
import turnelib as TE
import jclass as JC

def MyColor(color):
    if color == 'auto':
        return 'white'
    else :
        return color

def layout_frame(id, frame):
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
    layout = [
        eg.Input(default_text=str(frame.frameid).zfill(2), size=(2,1), key=key1),
        eg.Checkbox('', default=frame.draw, key=key2),
        eg.Input(default_text=str(frame.xname), size=(35,1), key=key3),
        eg.Checkbox('', default=frame.xrangefix, key=key4),   
        eg.Input(default_text=str(frame.xmin), size=(7,1), key=key5),
        eg.Input(default_text=str(frame.xmax), size=(7,1), key=key6),
        eg.Input(default_text=str(frame.yname), size=(35,1), key=key7),
        eg.Checkbox('', default=frame.yrangefix, key=key8), 
        eg.Input(default_text=str(frame.ymin), size=(7,1), key=key9),
        eg.Input(default_text=str(frame.ymax), size=(7,1), key=key10),
    ]
    return layout

def layout_trace(id, trace, colors):
    key1 = '-id{}-'.format(id)
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
    diags = [trace.name]
    layout = [
        eg.Text(str(id).zfill(2), size=(2, 1), key=key1,font=('Courier', 16)),
        eg.Checkbox('', default=trace.visible, key=key2),
        eg.Input(default_text = str(trace.frame), key=key3, size=(2, 1)),
        eg.Combo(values=diags, default_value=trace.name, size=(15, 20), key=key4),
        eg.Input(default_text=trace.strshot, size=(7,1), key=key5),
        eg.Input(default_text=trace.vals, size=(8,1), key=key6),        
        eg.Combo(values=[trace.tags, 'use_ValName', 'use_shotnum', 'use_name'], default_value=trace.tags, size=(10, 4), key=key7),        
        eg.Input(default_text=str(trace.nskip), size=(3,1), key=key8),
        eg.Input(default_text=str(trace.scale), size=(4,1), key=key9),
        eg.Input(default_text=str(trace.offset), size=(4,1), key=key10),
        eg.Checkbox('', default=trace.xmask, key=key11),        
        eg.Input(default_text=str(trace.xmin), size=(5,1), key=key12),
        eg.Input(default_text=str(trace.xmax), size=(5,1), key=key13),
        eg.Checkbox('', default=trace.ymask, key=key14),        
        eg.Input(default_text=str(trace.ymin), size=(5,1), key=key15),
        eg.Input(default_text=str(trace.ymax), size=(5,1), key=key16),
        eg.Input(default_text=trace.color, size=(6,1), color=MyColor(trace.color),enable_events=True, key=key17),
        eg.ColorBrowse('picker'),
    ]
    return layout

if __name__ == '__main__':

    colors = ['black','blue','red','gold','green','orange', 'brown', 'cyan', 'magenta', 'auto']

    parser = argparse.ArgumentParser(
        prog='jview',
        usage='jview -j tmp.json',
        description='edf file visualization manager',
        add_help=True,
        )
    parser.add_argument(
        '--version', 
        action='version', 
        version='Ver1.0')
    parser.add_argument(
        '-j',
        '--json',
        action='store',
        type=str,
        default = None,
        metavar = 'jsonfile',
        help='file name for setting file'
    ) 
    args = parser.parse_args()
    
# set initial conditions

    work = JC.loadfromjson(args.json)

    tabs = []

    l_general = [
        [eg.Text('Device: '+work.device)],
        [eg.Text('datapath:'),
         eg.InputText(default_text=work.datapath, enable_events=True, size=(60,1), key='-datapath-'),
         eg.FolderBrowse('Browse')],
        [eg.Text('directory prefix:'),
         eg.InputText(default_text=work.dprf, size=(4,1), key='-dprf-'),
         eg.Text('file extend:'),
         eg.InputText(default_text=work.fext, size=(4,1), key='-fext-')],
        [eg.Text('json file:'),
         eg.InputText(default_text='save as/load from json file', size=(60,1), enable_events=True, key='-filename-'),
         eg.FolderBrowse('Browse')], 
        [eg.Button('Load',size=(6,1),key='-load-'),
         eg.Button('Save',size=(6,1),key='-save-')]
    ]
    l_preference = [
        [eg.Text('Shot:', size=(5,1)),
         eg.Input(default_text=str(work.shotnum), size=(6,1), key='-shotnum-', enable_events=True),
         eg.Checkbox('', default=work.multishot, key='-multishot-'),
         eg.Text('Multi-Shot mode')],
        [eg.Text('Plot Style:', size=(8,1)),
         eg.Combo(values=['TEVO','RPRO'], default_value=work.plotstyle, size=(6, 2), key='-plotstyle-'),
         eg.Text('Xunit:', size=(5,1)),
         eg.Input(default_text=work.xunit, size=(6,1), key='-xunit-')],
        [eg.Text('Frame: col'),
         eg.Input(default_text = str(work.ncol), key="-cols-", size=(3, 1)),  #default = 2
         eg.Text(' x row'),
         eg.Input(default_text = str(work.nrow), key="-rows-", size=(3, 1))],  #default = 4
        [eg.Text('Page Style:'),
         eg.Combo(values=['slide','paper'], default_value=work.pagestyle, size=(6, 2), key='-pagestyle-'),
         eg.Text('fontsize:'),
         eg.Input(default_text=work.fontsize, size=(6,1), key='-fontsize-')],
        [eg.Text('Title:'),
         eg.InputText(default_text=work.title, key="-title-", size=(32, 1)),
         eg.Checkbox('auto',default=work.autotitle, key='-autotitle-')],
    ]
        
    id = 0
    for i in range(JC.NTAB_FRAME):
        tab_layout1= [[
            eg.Text(' id', size=(3,1), font=('Arial',12)),
            eg.Text('on', size=(3,1), font=('Arial',12)),
            eg.Text('xname', size=(45,1), font=('Arial',12)),
            eg.Text('x fix-range [ xmin : xmax ]', size=(26,1), font=('Arial',12)),
            eg.Text('yname', size=(45,1), font=('Arial',12)),
            eg.Text('y fix-range [ ymin : ymax ]', size=(26,1), font=('Arial',12)),
        ]]
        for j in range(JC.NFRAME_TAB):
            tab_layout1.append(layout_frame(id+1, work.frames[id]))
            id = id + 1
        tabs.append(eg.Tab('frame({}-{})'.format(i*JC.NFRAME_TAB+1, (i+1)*JC.NFRAME_TAB),tab_layout1))

    id = 0
    for i in range(JC.NTAB_TRACE):
        tab_layout2= [[
            eg.Text('', size=(3,1), font=('Arial',10)),
            eg.Text('on', size=(2,1), font=('Arial',12)),
            eg.Text('frame', size=(6,1), font=('Arial',12)),
            eg.Text('name', size=(22,1), font=('Arial',12)),
            eg.Text('shot', size=(11,1), font=('Arial',12)),
            eg.Text('vals', size=(9,1), font=('Arial',12)),
            eg.Text('tags', size=(15,1), font=('Arial',12)),
            eg.Text('nskip', size=(5,1), font=('Arial',12)),
            eg.Text('scale', size=(6,1), font=('Arial',12)),
            eg.Text('offset', size=(6,1), font=('Arial',12)),
            eg.Text('Xmask [ xmin : xmax ]', size=(19,1), font=('Arial',12)),
            eg.Text('Ymask [ ymin : ymax ]', size=(19,1), font=('Arial',12)),
            eg.Text('color', size=(6,1), font=('Arial',12))
        ]]
        for j in range(JC.NTRACE_TAB):
            tab_layout2.append(layout_trace(id+1, work.traces[id], colors))
            id = id + 1
        tabs.append(eg.Tab('trace({}-{})'.format(i*JC.NTRACE_TAB+1, (i+1)*JC.NTRACE_TAB),tab_layout2))

        
    layout = [
        [eg.Frame('General Setting',l_general), eg.Frame('Plot Preference', l_preference)],
        [eg.TabGroup([tabs])],
        [eg.Output(size=(130,3),key='-out-')],
        [eg.Button('Update',size=(12,1),key='-update-'), eg.Button('Plot',size=(12,1),key='-plot-'), eg.Button('Quit',size=(12,1),key='-quit-')]
    ]
            
    # Make Window

    window = eg.Window('jview', layout, font=('Arial',16), size=(1150, 750))

    subps = []
    while True :
        event, values = window.read()
        if event == '-shotnum-':
            if values['-multishot-'] :
                continue
            else:
                for t in work.traces:
                    id = t.id
                    key = '-strshot{}-'.format(id)   
                    window[key].update(values['-shotnum-'])
                continue
                
        for t in work.traces:
            key = '-color{}-'.format(t.id)
            if event == key :
                color = values[key]
                window[key].update(color=MyColor(color))
                break
            
        if event == '-update-':
            src_dir = values['-datapath-'] + values['-dprf-'] + values['-shotnum-']
            diags = JS.getdiagnames(src_dir)
            for t in work.traces :
                key = '-name{}-'.format(t.id)  
                val = values[key]
                window[key].update(values=diags)
            window['-out-'].print('Diagnames updated')
                  
        if event == '-load-':
            filename = values['-filename-']
            work = JC.loadfromjson(filename)
            window['-dprf-'].update(work.dprf)
            window['-fext-'].update(work.fext)            
            window['-datapath-'].update(work.datapath)
            window['-multishot-'].update(work.multishot)
            window['-shotnum-'].update(str(work.shotnum))            
            window['-plotstyle-'].update(work.plotstyle)
            window['-xunit-'].update(work.xunit)  
            window['-cols-'].update(str(work.ncol))
            window['-rows-'].update(str(work.nrow))
            window['-pagestyle-'].update(work.pagestyle)
            window['-fontsize-'].update(work.fontsize)
            window['-title-'].update(work.title)
            window['-autotitle-'].update(work.autotitle)
            for f in work.frames:
                id = f.id
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

                window[key1].update(f.frameid)
                window[key2].update(f.draw)
                window[key3].update(f.xname)
                window[key4].update(f.xrangefix)
                window[key5].update(str(f.xmin))
                window[key6].update(str(f.xmax))
                window[key7].update(f.yname)
                window[key8].update(f.yrangefix)
                window[key9].update(str(f.ymin))
                window[key10].update(str(f.ymax))
                
            for t in work.traces:
                id = t.id 
                key1 = '-id{}-'.format(id)
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

                window[key2].update(t.visible)
                window[key3].update(str(t.frame))
                window[key4].update(t.name)
                window[key5].update(t.strshot)
                window[key6].update(t.vals)
                window[key7].update(t.tags)
                window[key8].update(str(t.nskip))
                window[key9].update(str(t.scale))
                window[key10].update(str(t.offset))
                window[key11].update(t.xmask)
                window[key12].update(str(t.xmin))
                window[key13].update(str(t.xmax))
                window[key14].update(t.ymask)
                window[key15].update(str(t.ymin))
                window[key16].update(str(t.ymax))
                window[key17].update(t.color,color=MyColor(t.color))
            window['-out-'].print(filename+' loaded')
            
        if event == '-save-':
            print(values['-filename-'])
            filename = values['-filename-']
            work.update(values)
            JC.saveasjson(work, filename)
            window['-out-'].print(filename+' saved')

        if event == '-plot-':
            filename = 'tmp_jview.json'
            work.update(values)
            JC.saveasjson(work, filename)
            subp = subprocess.Popen(['python', 'jgraph.py', '-j', filename])
            subps.append(subp)    
         
        if event in (None, '-quit-'):
            for subp in subps:
                subp.kill()
            break             
        
    window.close()
    
