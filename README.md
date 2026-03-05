# ec-st-gnn-airpollution-forecasting
\documentclass[12pt, a4paper]{article}

% Packages
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{enumitem}

% Page Geometry
\geometry{a4paper, margin=1in}

% Title Configuration
\title{\textbf{Explainable Causal ST-GNN Digital Twin \\ for Real-Time Air Pollution Forecasting \\ and Source Attribution}}
\author{\textbf{Status:} Research Project \\ \textbf{License:} MIT (Recommended)}
\date{}

\begin{document}

\maketitle

\section*{Overview}
Current Air Quality Index (AQI) systems are largely reactive—they display current pollution levels but fail to predict future spikes, identify root causes, or simulate mitigation policies. Authorities are left without the intelligent tools necessary for effective, proactive intervention.

This project introduces a \textbf{City Digital Twin for Air Pollution}, a dynamic, virtual replica of a real-world urban environment. By modeling the city as a dynamic graph and leveraging \textbf{Spatio-Temporal Graph Neural Networks (ST-GNN)} alongside \textbf{Causal AI} and \textbf{Reinforcement Learning}, this system not only predicts future AQI with quantified confidence but also learns the cause-and-effect relationships driving pollution.

\section*{Main Objectives}
\begin{itemize}[label=\checkmark]
    \item \textbf{Accurate Forecasting:} Predict AQI at multiple horizons (1h, 3h, 6h ahead).
    \item \textbf{Source Attribution:} Pinpoint exact pollution source contributions (e.g., traffic vs. weather).
    \item \textbf{Causal Discovery:} Learn complex, real-world causal relationships between environmental variables.
    \item \textbf{Counterfactual Simulation:} Run "what-if" scenarios for proposed policies.
    \item \textbf{Policy Recommendation:} Recommend optimal mitigation strategies using Reinforcement Learning.
    \item \textbf{Uncertainty Estimation:} Provide confidence intervals for all predictions.
\end{itemize}

\section*{How We Model the City (Graph Representation)}
We map the physical city into a computational graph that continuously learns from real-time data:
\begin{itemize}
    \item \textbf{Nodes:} Air Quality Monitoring Stations
    \item \textbf{Edges:} Geographic proximity, wind-direction flow, and road connectivity
    \item \textbf{Node Features:}
    \begin{itemize}
        \item Pollutants: PM2.5, PM10, NO$_2$
        \item Meteorology: Temperature, Humidity, Wind speed, Wind direction
        \item Urban Data: Traffic intensity proxy
    \end{itemize}
\end{itemize}

\section*{System Architecture}
Our Digital Twin is powered by six interconnected modules:

\vspace{0.5em}
\noindent
\begin{tabularx}{\textwidth}{@{} l X X @{}}
\toprule
\textbf{Module} & \textbf{Core Technology} & \textbf{Function \& Output} \\
\midrule
\textbf{1. Spatio-Temporal Forecasting} & Graph Attention Network (GAT) + Temporal Attention & Models spatial dependencies and time-series data to output \textbf{future AQI values}. \\
\addlinespace
\textbf{2. Source Attribution} & Attention Weights Mapping & Groups features (traffic, weather) and outputs \textbf{percentage contribution}. \\
\addlinespace
\textbf{3. Causal Discovery} & Causal Inference (NOTEARS / PCMCI) & Discovers cause-effect links, outputting a \textbf{Causal Graph} explaining pollution dynamics. \\
\addlinespace
\textbf{4. Counterfactual Simulation} & Causal Modeling & Simulates "what-if" interventions to predict \textbf{how AQI would change}. \\
\addlinespace
\textbf{5. RL Policy Layer} & Reinforcement Learning Agent & Learns optimal strategies, rewarding AQI reduction. Outputs \textbf{Recommended Policy}. \\
\addlinespace
\textbf{6. Uncertainty Estimation} & Monte Carlo Dropout & Provides a confidence interval for predictions (e.g., AQI = 170 $\pm$ 12). \\
\bottomrule
\end{tabularx}
\vspace{1em}

\section*{Evaluation \& Metrics}
The system's performance is rigorously tested across three primary domains:

\subsection*{Forecasting Metrics}
\begin{itemize}
    \item \textbf{RMSE} (Root Mean Square Error)
    \item \textbf{MAE} (Mean Absolute Error)
    \item \textbf{R$^2$} (Coefficient of Determination)
\end{itemize}

\subsection*{Causal \& Explainability Metrics}
\begin{itemize}
    \item Counterfactual Consistency
    \item Causal Discovery Accuracy
\end{itemize}

\subsection*{Policy Simulation Metrics}
\begin{itemize}
    \item AQI Reduction Percentage
    \item Stability (Robustness under weather variations)
\end{itemize}

\section*{Why This Is Strong}
This project represents a \textbf{research-level contribution} to smart city governance. By utilizing real-world data, it moves beyond theoretical models into practical applicability. 

\textbf{Key Innovations:}
\begin{itemize}
    \item \textbf{Convergence of AI Fields:} Seamlessly merges GNNs, Causal AI, Explainable AI (XAI), and RL.
    \item \textbf{Actionable Intelligence:} Doesn't just predict a number; it tells policymakers \textit{why} it's happening and \textit{what} to do about it.
\end{itemize}

\section*{Social Impact}
\begin{itemize}
    \item \textbf{Pollution Control Boards:} Empowers authorities with proactive dashboards.
    \item \textbf{Traffic Management:} Enables data-driven, dynamic traffic routing.
    \item \textbf{Urban Planning:} Supports sustainable infrastructure development.
    \item \textbf{Public Health:} Protects vulnerable populations through early-warning systems.
\end{itemize}

\end{document}
