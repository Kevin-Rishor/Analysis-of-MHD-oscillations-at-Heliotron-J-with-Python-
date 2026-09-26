
tex = r'''\documentclass[11pt]{article}
\usepackage{geometry}
\geometry{a4paper, margin=1in}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{hyperref}

\title{Final Reconciliation Report: Shot 88653 Analysis}
\author{Antigravity Pipeline}
\date{\today}

\begin{document}

\maketitle

\section{Objective 1: Mode Identification}
\begin{table}[h!]
    \centering
    \begin{tabularx}{\textwidth}{l X X X l}
        \toprule
        \textbf{Claim} & \textbf{Deck Value} & \textbf{Fresh-run Value} & \textbf{General Summary} & \textbf{Status} \\
        \midrule
        Dominant Frequency & 89.0 kHz & 89.0 kHz & 89.0 kHz & Confirmed \\
        Toroidal Mode ($) & -5 ($\equiv -1$) & -5 & -5 & Confirmed \\
        Inter-probe Coherence & 0.86--0.88 (all pairs) & 0.866, 0.923, 0.942 & 0.866, 0.923, 0.942 & Needs deck update \\
        ECE-MP1 Coherence & $\sim.06 band-int (ECE12/13) & 0.187 bin (ECE03) & 0.187 bin (ECE03) & Needs deck update \\
        HAFAST Coupling & $\gamma^2 = 0.23$ & $\gamma^2 = 0.2335$ & $\gamma^2 = 0.2335$ & Confirmed \\
        \bottomrule
    \end{tabularx}
\end{table}

\textbf{Specific Resolution (Inter-probe Coherence):} The deck currently states `0.86--0.88 across all Mirnov pairs''. However, in the fresh run using the exact 259.1--275.0 ms window, only MP1-MP3 yields 0.866. MP1-MP4 (0.923) and MP3-MP4 (0.942) are significantly higher. The deck relies on the lowest pair and should report the full range (0.86--0.94).

\textbf{Specific Resolution (ECE-MP1 Coherence):} The deck cites ECE12/13 with band-integrated (80--100 kHz) coherence ($\sim.06). The fresh run confirms this band-integrated value (0.056 for ECE12, 0.064 for ECE13). However, evaluating single-bin coherence exactly at 89 kHz reveals that ECE03FAST is the peak channel ($\gamma^2 = 0.187$). Because the mode is narrowband, band-integration artificially dilutes the coherence with noise. The deck should be updated to cite the single-bin metric for ECE03FAST.

\section{Objective 2: Macroscopic Correlations (80--120 kHz HF Band)}
\begin{table}[h!]
    \centering
    \begin{tabularx}{\textwidth}{l X X X l}
        \toprule
        \textbf{Claim} & \textbf{Deck Value} & \textbf{Fresh-run Value} & \textbf{General Summary} & \textbf{Status} \\
        \midrule
        Envelope vs $ & 0.758 & 0.758 & 0.758 & Confirmed \\
        Envelope vs $ & 0.784 & 0.784 & 0.784 & Confirmed \\
        Envelope vs $ & 0.867 & 0.867 & 0.867 & Confirmed \\
        Envelope vs H$\alpha$ & 0.568 & 0.568 & 0.568 & Confirmed \\
        Validation Status & `correlates strongly'' & MEETS & MEETS & Confirmed \\
        \bottomrule
    \end{tabularx}
\end{table}

\textbf{Specific Resolution (Validation Status):} The fresh pipeline explicitly marks $, $, and $ as `MEETS'' because their {eff}$-adjusted Pearson p-values fall below 0.05. However, these are highly autocorrelated macroscopic variables. The deck's choice to label them as `correlates strongly'' rather than formally validated is appropriately conservative and should not be changed.

\section{Objective 2: Frequency Scaling Models}
\begin{table}[h!]
    \centering
    \begin{tabularx}{\textwidth}{l X X X l}
        \toprule
        \textbf{Claim} & \textbf{Deck Value} & \textbf{Fresh-run Value} & \textbf{General Summary} & \textbf{Status} \\
        \midrule
        LF Best Model & Current-induced ($) & Current-induced ($) & Current-induced ($) & Confirmed \\
        LF Model $|r|$ & 0.846 & 0.8465 & 0.8465 & Confirmed \\
        LF Model adj.p & 0.164 & 0.164 & 0.164 & Confirmed \\
        HF Best Model & BAE Acoustic/GAM & BAE Acoustic/GAM & BAE Acoustic/GAM & Confirmed \\
        HF Model $|r|$ & 0.479 & 0.4789 & 0.4789 & Confirmed \\
        HF Model adj.p & 0.204 & 0.204 & 0.204 & Confirmed \\
        \bottomrule
    \end{tabularx}
\end{table}

\section{Objective 3: Non-linear Coupling (Bicoherence)}
\begin{table}[h!]
    \centering
    \begin{tabularx}{\textwidth}{l X X X l}
        \toprule
        \textbf{Claim} & \textbf{Deck Value} & \textbf{Fresh-run Value} & \textbf{General Summary} & \textbf{Status} \\
        \midrule
        Headline (98.6/48.8) PSD & Not mentioned & NOT confirmed & NOT confirmed & Needs deck update \\
        Triad \#3 (89.8/41.0) PSD & Not mentioned & CONFIRMED & CONFIRMED & Needs deck update \\
        \bottomrule
    \end{tabularx}
\end{table}

\textbf{Specific Resolution (Triad PSD Match):} The fresh run confirms that the headline triad (98.6/48.8 kHz, ^2=0.515$) does not match real local PSD peaks (off by $>1.5$ bins). Meanwhile, Triad \#3 (89.8/41.0 kHz, ^2=0.328$) is perfectly grounded in confirmed spectral peaks. The headline choice (98.6/48.8) remains the mathematical maximum ^2$, but Triad \#3 deserves to be co-reported because of its physical grounding in validated PSD peaks.

\section{Objective 3: Validation}
\begin{table}[h!]
    \centering
    \begin{tabularx}{\textwidth}{l X X X l}
        \toprule
        \textbf{Claim} & \textbf{Deck Value} & \textbf{Fresh-run Value} & \textbf{General Summary} & \textbf{Status} \\
        \midrule
        Double-validation & Not passed & FAIL & FAIL & Confirmed \\
        Max ^2$ Quiescent & 0.37 & 0.366 & 0.366 & Confirmed \\
        Max ^2$ Control & 0.46 & 0.459 & 0.459 & Confirmed \\
        \bottomrule
    \end{tabularx}
\end{table}

\section{Proposed Deck Changes (Priority Order)}
\begin{enumerate}
    \item \textbf{Slide 10 (Objective 3):} Co-report Triad \#3 (89.8/41.0 kHz, ^2=0.328$) alongside the headline triad, noting that unlike the headline, Triad \#3 matches confirmed local PSD peaks.
    \item \textbf{Slide 5 (Objective 1):} Update ECE-MP1 coherence to cite ECE03FAST (0.19) evaluated precisely at the 89 kHz single bin, rather than the diluted band-integrated ECE12/13 value (0.06).
    \item \textbf{Slide 4 (Objective 1):} Correct the inter-probe coherence statement to `0.86--0.94'' to accurately encompass MP1-MP4 (0.92) and MP3-MP4 (0.94).
\end{enumerate}

\end{document}
'''
with open(r'c:\TFG\scratch\final_reconciliation.tex', 'w', encoding='utf-8') as f:
    f.write(tex)
