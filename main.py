#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programme de calcul de la fréquence de rotation (RPM) d'un moteur 2 temps
à partir d'un enregistrement audio (.mp3 ou .wav).

Fonctionnalités :
- Sélection du fichier audio via une interface graphique.
- Traitement du signal (filtrage, FFT, détection de pics).
- Affichage des résultats sous forme de graphiques :
  1. RPM par seconde (courbe temporelle).
  2. RPM moyen par plage de 15 secondes (histogramme).
  3. Pareto des durées cumulées par RPM.

Auteurs : Vibe Code (pour FloSigo)
Date : 2025
"""

import os
import sys
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import soundfile as sf
from scipy import signal
from scipy.fft import fft
from scipy.signal import find_peaks, butter, filtfilt
import pandas as pd


# =============================================================================
# PARAMÈTRES PAR DÉFAUT (adaptables via l'interface)
# =============================================================================
DEFAULT_NUM_CYLINDERS = 2  # Moteur 2 cylindres 2T
DEFAULT_LOW_CUT = 15.0     # Fréquence basse du filtre (Hz)
DEFAULT_HIGH_CUT = 150.0   # Fréquence haute du filtre (Hz)
DEFAULT_WINDOW_SIZE = 1.5  # Taille de la fenêtre FFT (secondes)
DEFAULT_HOP_LENGTH = 0.25  # Recouvrement entre fenêtres (secondes)
DEFAULT_BIN_SIZE = 15      # Taille des plages pour l'histogramme (secondes)
DEFAULT_PEAK_THRESHOLD = 0.1  # Seuil relatif pour la détection des pics (% du max)
DEFAULT_SMOOTH_WINDOW = 5  # Taille de la fenêtre de lissage (points)


# =============================================================================
# FONCTIONS DE TRAITEMENT DU SIGNAL
# =============================================================================

def load_audio_file(file_path: str, sr: int = None) -> tuple:
    """
    Charge un fichier audio (.mp3 ou .wav) et retourne le signal et le taux d'échantillonnage.
    
    Args:
        file_path (str): Chemin vers le fichier audio.
        sr (int, optional): Taux d'échantillonnage cible. Si None, utilise le taux natif.
    
    Returns:
        tuple: (signal mono, taux_échantillonnage)
    """
    try:
        # Utiliser librosa pour le MP3 (meilleure compatibilité)
        y, sr_orig = librosa.load(file_path, sr=sr, mono=True)
        return y, sr_orig
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement du fichier {file_path}: {e}")


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    """
    Applique un filtre passe-bande sur le signal.
    
    Args:
        data (np.ndarray): Signal à filtrer.
        lowcut (float): Fréquence de coupure basse (Hz).
        highcut (float): Fréquence de coupure haute (Hz).
        fs (float): Taux d'échantillonnage (Hz).
        order (int): Ordre du filtre.
    
    Returns:
        np.ndarray: Signal filtré.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def detect_dominant_frequency(y: np.ndarray, fs: float, low_freq: float, high_freq: float, 
                              peak_threshold: float = 0.1) -> float:
    """
    Détecte la fréquence dominante dans un signal via FFT.
    
    Args:
        y (np.ndarray): Signal temporel.
        fs (float): Taux d'échantillonnage (Hz).
        low_freq (float): Fréquence minimale à considérer (Hz).
        high_freq (float): Fréquence maximale à considérer (Hz).
        peak_threshold (float): Seuil relatif pour la détection des pics (0-1).
    
    Returns:
        float: Fréquence dominante (Hz). Retourne 0.0 si aucun pic détecté.
    """
    n = len(y)
    fft_result = np.abs(fft(y))
    freqs = np.linspace(0, fs, n)
    
    # Limiter à la plage de fréquences d'intérêt
    mask = (freqs >= low_freq) & (freqs <= high_freq)
    fft_filtered = fft_result[mask]
    freqs_filtered = freqs[mask]
    
    if len(freqs_filtered) == 0:
        return 0.0
    
    # Détecter les pics
    peaks, _ = find_peaks(fft_filtered, height=np.max(fft_filtered) * peak_threshold)
    
    if len(peaks) == 0:
        return 0.0
    
    # Retourner la fréquence du pic le plus intense
    peak_idx = peaks[np.argmax(fft_filtered[peaks])]
    return freqs_filtered[peak_idx]


def compute_rpm_from_audio(y: np.ndarray, fs: float, num_cylinders: int = 1,
                           low_cut: float = 15.0, high_cut: float = 150.0,
                           window_size: float = 1.5, hop_length: float = 0.25,
                           peak_threshold: float = 0.1) -> tuple:
    """
    Calcule les RPM à partir d'un signal audio en utilisant une FFT glissante.
    
    Args:
        y (np.ndarray): Signal audio mono.
        fs (float): Taux d'échantillonnage (Hz).
        num_cylinders (int): Nombre de cylindres (pour un 2T, 1 cycle = 1 tour).
        low_cut (float): Fréquence basse du filtre (Hz).
        high_cut (float): Fréquence haute du filtre (Hz).
        window_size (float): Taille de la fenêtre FFT (secondes).
        hop_length (float): Recouvrement entre fenêtres (secondes).
        peak_threshold (float): Seuil relatif pour la détection des pics.
    
    Returns:
        tuple: (times, rpm_values) où times est en secondes et rpm_values en tr/min.
    """
    # Convertir les paramètres en échantillons
    n_window = int(window_size * fs)
    n_hop = int(hop_length * fs)
    
    # Initialiser les listes de résultats
    times = []
    rpm_values = []
    
    # Appliquer le filtre passe-bande
    y_filtered = bandpass_filter(y, low_cut, high_cut, fs)
    
    # Parcourir le signal par fenêtres
    for i in range(0, len(y_filtered) - n_window, n_hop):
        window = y_filtered[i:i + n_window]
        time = i / fs  # Temps en secondes
        
        # Détecter la fréquence dominante
        dominant_freq = detect_dominant_frequency(window, fs, low_cut, high_cut, peak_threshold)
        
        # Convertir en RPM (pour un 2T : RPM = fréquence * 60 / 1)
        rpm = (dominant_freq * 60) / 1 if dominant_freq > 0 else 0.0
        
        times.append(time)
        rpm_values.append(rpm)
    
    return np.array(times), np.array(rpm_values)


def smooth_rpm(rpm_values: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Lisse les valeurs de RPM avec une moyenne mobile.
    
    Args:
        rpm_values (np.ndarray): Valeurs de RPM à lisser.
        window_size (int): Taille de la fenêtre de lissage.
    
    Returns:
        np.ndarray: Valeurs de RPM lissées.
    """
    if len(rpm_values) < window_size:
        return rpm_values
    
    kernel = np.ones(window_size) / window_size
    return np.convolve(rpm_values, kernel, mode='same')


def compute_rpm_bins(times: np.ndarray, rpm_values: np.ndarray, bin_size: float = 15.0) -> tuple:
    """
    Calcule les RPM moyens par plages de temps (bins).
    
    Args:
        times (np.ndarray): Tableau des temps (secondes).
        rpm_values (np.ndarray): Tableau des RPM.
        bin_size (float): Taille des plages (secondes).
    
    Returns:
        tuple: (bin_centers, bin_rpm_means, bin_counts)
    """
    bins = np.arange(0, np.max(times) + bin_size, bin_size)
    bin_indices = np.digitize(times, bins) - 1
    
    bin_rpm_sums = []
    bin_counts = []
    
    for i in range(len(bins) - 1):
        mask = (bin_indices == i)
        if np.any(mask):
            bin_rpm_sums.append(np.sum(rpm_values[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_rpm_sums.append(0.0)
            bin_counts.append(0)
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_rpm_means = np.array(bin_rpm_sums) / np.maximum(np.array(bin_counts), 1)
    
    return bin_centers, bin_rpm_means, np.array(bin_counts)


def compute_rpm_pareto(rpm_values: np.ndarray, times: np.ndarray = None) -> pd.DataFrame:
    """
    Calcule le Pareto des durées cumulées par RPM.
    
    Args:
        rpm_values (np.ndarray): Tableau des RPM.
        times (np.ndarray, optional): Tableau des temps. Si None, suppose un échantillonnage régulier.
    
    Returns:
        pd.DataFrame: DataFrame avec les RPM, durées, et durées cumulées.
    """
    if times is None:
        times = np.arange(len(rpm_values))
    
    # Discrétiser les RPM en plages de 50 (pour éviter trop de valeurs uniques)
    rpm_bins = np.arange(0, np.max(rpm_values) + 50, 50)
    bin_indices = np.digitize(rpm_values, rpm_bins) - 1
    
    # Calculer la durée passée dans chaque plage
    bin_durations = []
    for i in range(len(rpm_bins) - 1):
        mask = (bin_indices == i)
        if np.any(mask):
            # Durée = somme des intervalles de temps pour cette plage
            duration = np.sum(np.diff(times[mask])) if len(times[mask]) > 1 else 0.0
            bin_durations.append(duration)
        else:
            bin_durations.append(0.0)
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'RPM_min': rpm_bins[:-1],
        'RPM_max': rpm_bins[1:],
        'RPM_moyen': (rpm_bins[:-1] + rpm_bins[1:]) / 2,
        'Durée (s)': bin_durations
    })
    
    # Supprimer les plages sans durée
    df = df[df['Durée (s)'] > 0].copy()
    
    # Calculer les durées cumulées
    df['Durée cumulée (s)'] = df['Durée (s)'].cumsum()
    df['% cumulé'] = (df['Durée cumulée (s)'] / df['Durée (s)'].sum()) * 100
    
    # Trier par durée décroissante
    df = df.sort_values('Durée (s)', ascending=False).reset_index(drop=True)
    
    return df


# =============================================================================
# CLASSE PRINCIPALE DE L'INTERFACE GRAPHIQUE
# =============================================================================

class RPMAnalyzerApp:
    """Application principale pour l'analyse des RPM à partir d'un fichier audio."""
    
    def __init__(self, root: tk.Tk):
        """
        Initialise l'application.
        
        Args:
            root (tk.Tk): Racine de l'interface Tkinter.
        """
        self.root = root
        self.root.title("Analyseur de RPM - Moteur 2 Temps")
        self.root.geometry("1200x900")
        
        # Variables de configuration
        self.file_path = tk.StringVar()
        self.num_cylinders = tk.IntVar(value=DEFAULT_NUM_CYLINDERS)
        self.low_cut = tk.DoubleVar(value=DEFAULT_LOW_CUT)
        self.high_cut = tk.DoubleVar(value=DEFAULT_HIGH_CUT)
        self.window_size = tk.DoubleVar(value=DEFAULT_WINDOW_SIZE)
        self.hop_length = tk.DoubleVar(value=DEFAULT_HOP_LENGTH)
        self.bin_size = tk.DoubleVar(value=DEFAULT_BIN_SIZE)
        self.peak_threshold = tk.DoubleVar(value=DEFAULT_PEAK_THRESHOLD)
        self.smooth_window = tk.IntVar(value=DEFAULT_SMOOTH_WINDOW)
        
        # Données calculées
        self.times = None
        self.rpm_values = None
        self.smoothed_rpm = None
        self.bin_centers = None
        self.bin_rpm_means = None
        self.bin_counts = None
        self.pareto_df = None
        
        # Créer l'interface
        self.create_widgets()
    
    def create_widgets(self):
        """Crée les widgets de l'interface."""
        # Frame pour les contrôles
        control_frame = ttk.LabelFrame(self.root, text="Paramètres", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Bouton pour sélectionner le fichier
        ttk.Label(control_frame, text="Fichier audio:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(control_frame, textvariable=self.file_path, width=50).grid(row=0, column=1, sticky=tk.W, pady=2)
        ttk.Button(control_frame, text="Parcourir...", command=self.browse_file).grid(row=0, column=2, padx=5, pady=2)
        
        # Paramètres du moteur
        ttk.Label(control_frame, text="Nombre de cylindres:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(control_frame, from_=1, to=10, textvariable=self.num_cylinders, width=5).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # Paramètres du filtre
        ttk.Label(control_frame, text="Fréquence basse (Hz):").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(control_frame, textvariable=self.low_cut, width=10).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(control_frame, text="Fréquence haute (Hz):").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Entry(control_frame, textvariable=self.high_cut, width=10).grid(row=3, column=1, sticky=tk.W, pady=2)
        
        # Paramètres de la FFT
        ttk.Label(control_frame, text="Taille fenêtre FFT (s):").grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Entry(control_frame, textvariable=self.window_size, width=10).grid(row=4, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(control_frame, text="Recouvrement (s):").grid(row=5, column=0, sticky=tk.W, pady=2)
        ttk.Entry(control_frame, textvariable=self.hop_length, width=10).grid(row=5, column=1, sticky=tk.W, pady=2)
        
        # Paramètres de détection
        ttk.Label(control_frame, text="Seuil pics (% max):").grid(row=6, column=0, sticky=tk.W, pady=2)
        ttk.Entry(control_frame, textvariable=self.peak_threshold, width=10).grid(row=6, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(control_frame, text="Lissage (points):").grid(row=7, column=0, sticky=tk.W, pady=2)
        ttk.Spinbox(control_frame, from_=1, to=20, textvariable=self.smooth_window, width=5).grid(row=7, column=1, sticky=tk.W, pady=2)
        
        # Paramètres des plages
        ttk.Label(control_frame, text="Taille plages (s):").grid(row=8, column=0, sticky=tk.W, pady=2)
        ttk.Entry(control_frame, textvariable=self.bin_size, width=10).grid(row=8, column=1, sticky=tk.W, pady=2)
        
        # Bouton d'analyse
        ttk.Button(control_frame, text="Analyser", command=self.run_analysis, style="Accent.TButton").grid(
            row=9, column=0, columnspan=3, pady=10)
        
        # Frame pour les graphiques
        self.graph_frame = ttk.Frame(self.root)
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialiser les graphiques
        self.create_graphs()
    
    def create_graphs(self):
        """Initialise les 3 graphiques."""
        # Graphique 1 : RPM par seconde
        self.fig1, self.ax1 = plt.subplots(figsize=(5, 3))
        self.ax1.set_title("RPM par seconde")
        self.ax1.set_xlabel("Temps (s)")
        self.ax1.set_ylabel("RPM")
        self.ax1.grid(True)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.graph_frame)
        self.canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Graphique 2 : RPM par plage de 15 secondes
        self.fig2, self.ax2 = plt.subplots(figsize=(5, 3))
        self.ax2.set_title("RPM moyen par plage de 15 secondes")
        self.ax2.set_xlabel("Temps (s)")
        self.ax2.set_ylabel("RPM moyen")
        self.ax2.grid(True)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.graph_frame)
        self.canvas2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Graphique 3 : Pareto des durées cumulées
        self.fig3, self.ax3 = plt.subplots(figsize=(5, 3))
        self.ax3.set_title("Pareto des durées cumulées par RPM")
        self.ax3.set_xlabel("RPM moyen")
        self.ax3.set_ylabel("Durée cumulée (%)")
        self.ax3.grid(True)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.graph_frame)
        self.canvas3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    def browse_file(self):
        """Ouvre une boîte de dialogue pour sélectionner un fichier."""
        file_path = filedialog.askopenfilename(
            title="Sélectionner un fichier audio",
            filetypes=[("Fichiers audio", "*.mp3 *.wav"), ("Tous les fichiers", "*.*")]
        )
        if file_path:
            self.file_path.set(file_path)
    
    def run_analysis(self):
        """Exécute l'analyse complète."""
        # Vérifier qu'un fichier est sélectionné
        if not self.file_path.get():
            messagebox.showerror("Erreur", "Veuillez sélectionner un fichier audio.")
            return
        
        try:
            # Charger le fichier audio
            self.root.config(cursor="wait")
            self.root.update()
            
            y, fs = load_audio_file(self.file_path.get())
            
            # Calculer les RPM
            times, rpm_values = compute_rpm_from_audio(
                y, fs,
                num_cylinders=self.num_cylinders.get(),
                low_cut=self.low_cut.get(),
                high_cut=self.high_cut.get(),
                window_size=self.window_size.get(),
                hop_length=self.hop_length.get(),
                peak_threshold=self.peak_threshold.get()
            )
            
            # Lisser les RPM
            self.smoothed_rpm = smooth_rpm(rpm_values, self.smooth_window.get())
            
            # Calculer les RPM par plages de 15 secondes
            self.bin_centers, self.bin_rpm_means, self.bin_counts = compute_rpm_bins(
                times, self.smoothed_rpm, self.bin_size.get()
            )
            
            # Calculer le Pareto
            self.pareto_df = compute_rpm_pareto(self.smoothed_rpm, times)
            
            # Mettre à jour les graphiques
            self.update_graphs(times, rpm_values, self.smoothed_rpm)
            
            self.root.config(cursor="")
            
        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("Erreur", f"Une erreur est survenue: {e}")
    
    def update_graphs(self, times: np.ndarray, rpm_values: np.ndarray, smoothed_rpm: np.ndarray):
        """Met à jour les 3 graphiques avec les nouvelles données."""
        # Graphique 1 : RPM par seconde
        self.ax1.clear()
        self.ax1.plot(times, rpm_values, alpha=0.3, label="RPM brut")
        self.ax1.plot(times, smoothed_rpm, label="RPM lissé", color="red")
        self.ax1.set_title("RPM par seconde")
        self.ax1.set_xlabel("Temps (s)")
        self.ax1.set_ylabel("RPM")
        self.ax1.legend()
        self.ax1.grid(True)
        self.canvas1.draw()
        
        # Graphique 2 : RPM par plage de 15 secondes
        self.ax2.clear()
        valid_bins = self.bin_counts > 0
        self.ax2.bar(
            self.bin_centers[valid_bins], 
            self.bin_rpm_means[valid_bins], 
            width=self.bin_size.get() * 0.8,
            align="center"
        )
        self.ax2.set_title(f"RPM moyen par plage de {self.bin_size.get()} secondes")
        self.ax2.set_xlabel("Temps (s)")
        self.ax2.set_ylabel("RPM moyen")
        self.ax2.grid(True)
        self.canvas2.draw()
        
        # Graphique 3 : Pareto des durées cumulées
        self.ax3.clear()
        if self.pareto_df is not None and not self.pareto_df.empty:
            self.ax3.bar(
                self.pareto_df['RPM_moyen'], 
                self.pareto_df['Durée (s)'], 
                width=40, 
                align="center",
                alpha=0.7,
                label="Durée par RPM"
            )
            self.ax3.plot(
                self.pareto_df['RPM_moyen'], 
                self.pareto_df['% cumulé'], 
                color="red", 
                marker="o", 
                label="% cumulé"
            )
            self.ax3.set_title("Pareto des durées cumulées par RPM")
            self.ax3.set_xlabel("RPM moyen")
            self.ax3.set_ylabel("Durée (s) / % cumulé")
            self.ax3.legend()
            self.ax3.grid(True)
        self.canvas3.draw()


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    # Vérifier les dépendances
    try:
        import librosa
        import soundfile
        import scipy
        import pandas
        import matplotlib
    except ImportError as e:
        messagebox.showerror(
            "Erreur de dépendances",
            f"Dépendance manquante: {e}.\n\nInstallez les dépendances avec:\npip install numpy scipy matplotlib librosa soundfile pandas"
        )
        sys.exit(1)
    
    # Créer l'application
    root = tk.Tk()
    app = RPMAnalyzerApp(root)
    
    # Lancer la boucle principale
    root.mainloop()
