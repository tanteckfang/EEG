# EEG Decode and Light Control with GUI and Hue Integration
# Author: Teck Fang | Purpose: Control Philips Hue lights using real-time EEG alpha power

import os
import time
import numpy as np
import pandas as pd
import requests
from scipy.signal import welch
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import tkinter as tk
from nbclient import NotebookClient
from nbformat import read as nb_read
import threading
import nest_asyncio

# Allow asyncio inside Jupyter/Tkinter context
nest_asyncio.apply()

# ---------------------------- CONFIGURATION ---------------------------- #
BRIDGE_IP = "192.168.226.220"  # IP address of the Hue Bridge
USERNAME = "ZqKauJLZ3x4je5RxtKnaTAK958AIYRBYmMHNqhEw"  # Authorized Hue API username
COLOR_SEQUENCES = {
    "Sequence 1": (0, 46920),       # Red to Blue
    "Sequence 2": (25500, 12750),   # Green to Yellow
    "Sequence 3": (46920, 56100),   # Blue to Purple
    "Sequence 4": (0, 56100),       # Red to Purple
}

SELECTED_SEQUENCE = "Sequence 1" # change according to your preference
baseline_alpha = 5e+01  # Default baseline alpha
SELECTED_LIGHTS = ["Light 1", "Light 2"]

FS = 250  # EEG Sampling rate (Hz)
WINDOW_LEN_S = 4  # Length of EEG window to analyze in seconds
smoothing_factor = 0.1  # Smoothing factor for exponential moving average

# ---------------------------- RUNTIME STATE ---------------------------- #
last_valid_metric = 0.5
baseline_alpha = None
recording_baseline = False
baseline_data = []
alpha_smooth = 0.0
included_channels = [0, 1, 2, 3]  # Default: all 4 EEG channels

# ---------------------------- HUE LIGHT CONTROL ---------------------------- #
def send_to_light(hue_val):
    """Send hue value to selected Philips Hue lights."""
    payload = {"on": True, "bri": 200, "hue": int(hue_val), "sat": 254}
    for name in SELECTED_LIGHTS:
        light_id = LIGHT_IDS.get(name)
        if light_id:
            url = f"http://{BRIDGE_IP}/api/{USERNAME}/lights/{light_id}/state"
            try:
                response = requests.put(url, json=payload, timeout=1)
                print(f"[HUE] Sent hue {int(hue_val)} to {name}. Response: {response.status_code}")
            except Exception as e:
                print(f"[HUE] Failed to send to light {name}: {e}")

# ---------------------------- EEG PROCESSING ---------------------------- #
def process_eeg_chunk(eeg_data):
    """Process EEG chunk: compute alpha power and send mapped hue to lights."""
    global last_valid_metric, baseline_alpha, recording_baseline, baseline_data, alpha_smooth

    # Select only included EEG channels
    eeg_data = eeg_data[:, included_channels]

    # Collect data if recording baseline
    if recording_baseline:
        baseline_data.append(eeg_data)
        print("[BASELINE] Collecting baseline data...")
        return

    # Compute PSD and extract alpha power (8-13 Hz)
    freqs, psd = welch(eeg_data.mean(axis=1), fs=FS, nperseg=FS)
    alpha_band = (freqs >= 8) & (freqs <= 13)
    alpha_power = np.trapz(psd[alpha_band], freqs[alpha_band])

    # Apply smoothing
    alpha_smooth = (1 - smoothing_factor) * alpha_smooth + smoothing_factor * alpha_power

    # Normalize against baseline (default 25 if not yet set)
    if baseline_alpha:
        metric = np.clip(alpha_smooth / baseline_alpha, 0, 1)
    else:
        metric = np.clip(alpha_smooth / 25.0, 0, 1)

    last_valid_metric = metric

    # Interpolate hue from metric
    hue_start, hue_end = COLOR_SEQUENCES[SELECTED_SEQUENCE]
    hue_val = (1 - metric) * hue_start + metric * hue_end

    # Display and send
    print(f"[EEG] Alpha Power: {alpha_power:.2e}, Smoothed: {alpha_smooth:.2e}, Normalized Metric: {metric:.2f}, Hue: {int(hue_val)}")
    status = "Below baseline alpha" if baseline_alpha and alpha_smooth < baseline_alpha else ""
    send_to_light(hue_val)
    update_gui(alpha_power, metric, hue_val, status)

# ---------------------------- FILE MONITORING ---------------------------- #
class FileHandler(FileSystemEventHandler):
    """Watches EEG recording folder and reacts to file changes."""
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def on_modified(self, event):
        if event.src_path.endswith(".txt"):
            try:
                print(f"[DEBUG] File modified: {event.src_path}")
                df = pd.read_csv(event.src_path, delimiter=',', skiprows=4, header=None)
                df.dropna(inplace=True)
                eeg_only = df.iloc[:, 1:5]
                if len(eeg_only) >= FS * WINDOW_LEN_S:
                    eeg_chunk = eeg_only.tail(FS * WINDOW_LEN_S).astype(float).values
                    print(f"[DEBUG] EEG chunk shape: {eeg_chunk.shape}")
                    process_eeg_chunk(eeg_chunk)
            except Exception as e:
                print(f"Error reading EEG file: {e}")

# ---------------------------- GUI DISPLAY ---------------------------- #
def update_gui(power, metric, hue_val, status=""):
    """Update GUI with current alpha power and status."""
    power_label.config(text=f"Alpha Power: {power:.2e}")
    metric_label.config(text=f"Metric: {metric:.2f}")
    hue_label.config(text=f"Hue Sent: {int(hue_val)}")
    status_label.config(text=status)
    if baseline_alpha:
        baseline_display_label.config(text=f"Baseline Alpha Power: {baseline_alpha:.2e}")
    meter_canvas.delete("meter")
    bar_height = int(metric * 100)
    color = "#00ff00" if metric < 0.33 else "#ffff00" if metric < 0.66 else "#ff0000"
    meter_canvas.create_rectangle(10, 100 - bar_height, 90, 100, fill=color, tags="meter")
    root.update_idletasks()

def start_baseline():
    """Initiate 12s baseline collection."""
    global recording_baseline, baseline_data
    recording_baseline = True
    baseline_data = []
    status_label.config(text="Recording baseline for 12 seconds...")
    root.after(12000, finish_baseline)

def finish_baseline():
    """Compute median alpha power over the 12s baseline and adjust by 75%."""
    global recording_baseline, baseline_alpha
    recording_baseline = False
    full_baseline = np.concatenate(baseline_data, axis=0)
    freqs, psd = welch(full_baseline.mean(axis=1), fs=FS, nperseg=FS)
    alpha_band = (freqs >= 8) & (freqs <= 13)
    raw_baseline = np.trapz(psd[alpha_band], freqs[alpha_band])
    baseline_alpha = 0.75 * np.median(raw_baseline)  # Lowered by 25%
    status_label.config(text=f"Baseline set: {baseline_alpha:.2e}")
    baseline_label.config(text=f"Baseline Alpha Power: {baseline_alpha:.2e}")
    baseline_display_label.config(text=f"Baseline Alpha Power: {baseline_alpha:.2e}")

# ---------------------------- NOTEBOOK INTEGRATION ---------------------------- #
def run_notebook():
    """Execute supporting EEG notebook automatically on startup."""
    def task():
        try:
            print("Executing EEG decoding notebook...")
            path_to_nb = os.path.join(os.getcwd(), "eeg_decode_teckfang.ipynb")
            with open(path_to_nb) as f:
                nb = nb_read(f, as_version=4)
                client = NotebookClient(nb, timeout=600)
                client.execute()
            print("Notebook execution complete.")
        except Exception as e:
            print(f"Notebook execution failed: {e}")
    threading.Thread(target=task, daemon=True).start()

# ---------------------------- MAIN LOOP ---------------------------- #
def main():
    run_notebook()
    print("Monitoring EEG files for updates...")
    base_folder_path = r"C:\\Users\\ttfta\\Documents\\OpenBCI_GUI\\Recordings" # to be changed based on where you save the OpenBCI_GUI recordings
    folder = "OpenBCISession_2025-06-15_22-45-49" # to be change everytime before a trial is start
    folder_path = os.path.join(base_folder_path, folder)
    event_handler = FileHandler(folder_path)
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()
    try:
        root.mainloop()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# ---------------------------- GUI SETUP ---------------------------- #
root = tk.Tk()
root.title("EEG Alpha Monitor")
root.geometry("300x500")

power_label = tk.Label(root, text="Alpha Power: --")
metric_label = tk.Label(root, text="Metric: --")
hue_label = tk.Label(root, text="Hue Sent: --")
baseline_label = tk.Label(root, text="Baseline Alpha Power: --")
baseline_display_label = tk.Label(root, text="Baseline Alpha Power: --")
status_label = tk.Label(root, text="Status: --")

for widget in [power_label, metric_label, hue_label, baseline_label, baseline_display_label, status_label]:
    widget.pack(pady=5)

tk.Button(root, text="Start Baseline", command=start_baseline).pack(pady=5)

# Channel selection checkboxes
channel_vars = []
for i in range(4):
    var = tk.IntVar(value=1)
    chk = tk.Checkbutton(root, text=f"Include Channel {i+1}", variable=var)
    chk.pack(anchor="w")
    channel_vars.append(var)

def update_channels():
    global included_channels
    included_channels = [i for i, var in enumerate(channel_vars) if var.get() == 1]
    print(f"[CONFIG] Included channels: {included_channels}")

tk.Button(root, text="Apply Channel Selection", command=update_channels).pack(pady=5)

# Real-time metric visual bar
meter_canvas = tk.Canvas(root, width=100, height=100)
meter_canvas.pack(pady=5)
meter_canvas.create_rectangle(10, 0, 90, 100, outline="black")

# ---------------------------- ENTRY POINT ---------------------------- #
if __name__ == "__main__":
    main()
