[2025-Nov-12] Version 1.1 Released

-----------------------------------
         Applications
-----------------------------------

anaspect.py
    This application calculates auto power spectrum of a time-series-data.
    The time-series-data should be given as "edf" format.
    Outputs are results of
         1. running FFT (short time FFT)
         2. Auto Power Spectrum which is ensemble averaged over a specified time period.

    Usage:
    anaaspect [-h] [--version] [-f filename] [-t id] -r fres [-w]

    optional arguments:
        -h, --help             Show this help message and exit
        --version              Show program's version number and exit
        -f filename, --file filename
                               Set an edf-formatted file to be read
        -t id, --target id     Set variables IDs to be plotted (default 1)
        -r fres, --resolution fres
                               Set frequency resolution in [Hz]
        -w, --write            Save results to files

    Example:
        anaaspect.py -f ../expdata/FY2022/MP3@83489.edf -r 500

    Dependence
    ----------
    turnelib.py
    libana_signal.py
    matplotlibutil.py

    Status
    ------
    Version 1.0

anahilbert.py
    This application calculates analysis signal of a time-series-data.
    The time-series-data should be given as "edf" format.
    Outputs are results of
         1. Band-passed signal
         2. Envelope of the band-passed signal
         3. Instantaneous frequency which is calculated from time-derivative of
            the instantaneous phase

    Usage:
    anahilbert [-h] [--version] [-f filename] [-t id] [-s sgwin] [-l fl]
                    [-u fu] [-o nord] [-w]


    optional arguments:
        -h, --help             show this help message and exit
        --version              show program's version number and exit
        -f filename, --file filename
                               Set an edf-formatted file to be read
        -t id, --target id     Set variables IDs to be plotted (default 1)
        -s sgwin, --smoothing sgwin
                               Set window width for savgol_filter (odd-integer)
        -l fl, --lower fl      Set lower frequency for band-pass-filter in [kHz]
        -u fu, --upper fu      Set upper frequency for band-pass-filter in [kHz]
        -o nord, --order nord
                               Set order for Bessel filter (default=4)
        -w, --write            Save results to files

    Example:
        python anahilbert.py -f ../expdata/FY2022/MP3@83489.edf 

    Dependence
    ----------
    turnelib.py
    libana_signal.py
    matplotlibutil.py

    Status
    ------
    Version 1.0


analpf.py
    This application applys the low-passed-filter to a time-series-data.
    The time-series-data should be given as "edf" format.

    Usage:
    analpf [-h] [--version] [-f filename] [-t id] [-c fc] [-o nord] [-w]


    optional arguments:
        -h, --help             show this help message and exit
        --version              show program's version number and exit
        -f filename, --file filename
                               Set an edf-formatted file to be read
        -t id, --target id     Set variables IDs to be plotted (default 1)
        -s sgwin, --smoothing sgwin
                               Set window width for savgol_filter (odd-integer)
        -c fc, --cut fc        Set cut-off frequency for low-pass-filter in [kHz]
        -w, --write            Save results to files

    Example:
        python analpf.py -f ../expdata/FY2022/MP3@83489.edf -c 1

    Dependence
    ----------
    turnelib.py
    libana_signal.py
    matplotlibutil.py

    Status
    ------
    Version 1.0


edfview.py
    Visualize an edf-format file by using matplotlib.
    igetfile like function is available for LHD data.

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
    Version 2.0


jcnv2to1.py
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


jgraph.py
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


jview.py
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


tsview.py
    This program visualize Thomson scattering measurement datawhich is saved as a edf-formatted file.

    Usage: tsview [-h] [--version] -f filename [-T id id id id] [-R [id ...]] [-t [id ...]] [-p] [-r]
                            [-M val val]

    options:
        -h, --help            Show this help message and exit
        --version             Show program's version number and exit
        -f filename, --file filename
                              Set an edf-formatted file to be read
        -T id id id id, --Thomson id id id id
                              Set IDs for [Te, dT, ne, dn]
        -R [id ...], --Rslices [id ...]
                              Set IDs of R for R-slices (default 1)
        -t [id ...], --timeslices [id ...]
                              Set IDs of time for time-slices (default 1)
       -p, --portrait         Portrait mode
       -r, --reverse          Reverse color map (default True jet_r)
       -M val val, --Max val val
                              Set [Te_mac, ne_max]

    Example:
        python tsview.py -f 194934/thomson@194934_1.txt -M 1.5 30 -T 1 2 3 4

    Dependence
    ----------
    turnelib.py
    matplotlibutil.py

    Status
    ------
    Version 1.0

-----------------------------------
         modules
-----------------------------------

turnelib.py 
    General purpose utility tools

matplotlibutil.py
    Plot utilities (matplotlib wrapper).

libana_signal.py
    General purpose signal processing tools

jclass.py
    Common classes and functions used across the jpack applications

jslang.py
    Device-dependent functions
