# EEG Decode and Light Control with GUI and Hue Integration
# Author: Teck Fang | Purpose: Control Philips Hue lights using real-time EEG alpha power

import os
import time
import numpy as np
import pandas as pd
import requests
from scipy.signal import welch
import tkinter as tk

# ---------------------------- CONFIGURATION ---------------------------- #
BRIDGE_IP = "192.168.226.220"  # Hue Bridge IP
USERNAME = "ZqKauJLZ3x4je5RxtKnaTAK958AIYRBYmMHNqhEw"
COLOR_SEQUENCES = {
    "Sequence 1": (0, 46920),       # Red to Blue
    "Sequence 2": (25500, 12750),   # Green to Yellow
    "Sequence 3": (46920, 56100),   # Blue to Purple
    "Sequence 4": (0, 56100),       # Red to Purple
}
SELECTED_SEQUENCE = "Sequence 1"
SELECTED_LIGHTS = ["Light 1", "Light 2"]
LIGHT_IDS = {"Light 1": 1, "Light 2": 2}

FS = 250
WINDOW_LEN_S = 4
smoothing_factor = 0.1

# ---------------------------- RUNTIME STATE ---------------------------- #
last_file_size = 0
last_valid_metric = 0.5
baseline_alpha = None
recording_baseline = False
baseline_data = []
alpha_smooth = 0.0
included_channels = [0, 1, 2, 3]

# ---------------------------- HUE LIGHT CONTROL ---------------------------- #
def send_to_light(hue_val):
    payload = {"on": True, "bri": 200, "hue": int(hue_val), "sat": 254}
    for name in SELECTED_LIGHTS:
        light_id = LIGHT_IDS.get(name)
        if light_id:
            url = f"http://{BRIDGE_IP}/api/{USERNAME}/lights/{light_id}/state"
            try:
                requests.put(url, json=payload, timeout=1)
            except Exception as e:
                print(f"[HUE ERROR] {e}")

# ---------------------------- EEG PROCESSING ---------------------------- #
def process_eeg_chunk(eeg_data):
    global last_valid_metric, baseline_alpha, recording_baseline, baseline_data, alpha_smooth
    eeg_data = eeg_data[:, included_channels]

    if recording_baseline:
        baseline_data.append(eeg_data)
        print("[BASELINE] Collecting baseline data...")
        return

    freqs, psd = welch(eeg_data.mean(axis=1), fs=FS, nperseg=FS)
    alpha_band = (freqs >= 8) & (freqs <= 13)
    alpha_power = np.trapz(psd[alpha_band], freqs[alpha_band])
    alpha_smooth = (1 - smoothing_factor) * alpha_smooth + smoothing_factor * alpha_power

    if baseline_alpha:
        metric = np.clip(alpha_smooth / baseline_alpha, 0, 1)
    else:
        metric = np.clip(alpha_smooth / 25.0, 0, 1)

    last_valid_metric = metric
    hue_start, hue_end = COLOR_SEQUENCES[SELECTED_SEQUENCE]
    hue_val = (1 - metric) * hue_start + metric * hue_end

    print(f"[EEG] Alpha: {alpha_power:.2e} | Smooth: {alpha_smooth:.2e} | Metric: {metric:.2f}")
    status = "Below baseline alpha" if baseline_alpha and alpha_smooth < baseline_alpha else ""
    send_to_light(hue_val)
    update_gui(alpha_power, metric, hue_val, status)

# ---------------------------- GUI DISPLAY ---------------------------- #
def update_gui(power, metric, hue_val, status=""):
    power_label.config(text=f"Alpha Power: {power:.2e}")
    metric_label.config(text=f"Metric: {metric:.2f}")
    hue_label.config(text=f"Hue Sent: {int(hue_val)}")
    status_label.config(text=status)
    if baseline_alpha:
        baseline_display_label.config(text=f"Baseline Alpha: {baseline_alpha:.2e}")
    meter_canvas.delete("meter")
    bar_height = int(metric * 100)
    color = "#00ff00" if metric < 0.33 else "#ffff00" if metric < 0.66 else "#ff0000"
    meter_canvas.create_rectangle(10, 100 - bar_height, 90, 100, fill=color, tags="meter")
    root.update_idletasks()

def start_baseline():
    global recording_baseline, baseline_data
    recording_baseline = True
    baseline_data = []
    status_label.config(text="Recording baseline for 12 seconds...")
    root.after(12000, finish_baseline)

def finish_baseline():
    global recording_baseline, baseline_alpha
    recording_baseline = False
    full_baseline = np.concatenate(baseline_data, axis=0)
    freqs, psd = welch(full_baseline.mean(axis=1), fs=FS, nperseg=FS)
    alpha_band = (freqs >= 8) & (freqs <= 13)
    alpha_power = np.trapz(psd[alpha_band], freqs[alpha_band])
    baseline_alpha = 0.75 * np.median(alpha_power)
    status_label.config(text=f"Baseline set: {baseline_alpha:.2e}")
    baseline_display_label.config(text=f"Baseline Alpha: {baseline_alpha:.2e}")

# ---------------------------- FILE POLLING ---------------------------- #
def poll_latest_file():
    global last_file_size
    base_folder_path = r"C:\\Users\\ttfta\\Documents\\OpenBCI_GUI\\Recordings"
    folder = "OpenBCISession_2025-07-14_15-52-11"
    folder_path = os.path.join(base_folder_path, folder)

    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    if not txt_files:
        root.after(1000, poll_latest_file)
        return

    latest_file = max([os.path.join(folder_path, f) for f in txt_files], key=os.path.getmtime)
    size = os.path.getsize(latest_file)

    if size != last_file_size:
        last_file_size = size
        try:
            df = pd.read_csv(latest_file, delimiter=",", skiprows=4, header=None)
            df.dropna(inplace=True)
            eeg_only = df.iloc[:, 1:5]
            if len(eeg_only) >= FS * WINDOW_LEN_S:
                chunk = eeg_only.tail(FS * WINDOW_LEN_S).astype(float).values
                process_eeg_chunk(chunk)
        except Exception as e:
            print(f"[READ ERROR] {e}")
    else:
        print("[INFO] Waiting for new data...")

    root.after(1000, poll_latest_file)

# ---------------------------- GUI SETUP ---------------------------- #
root = tk.Tk()
root.title("EEG Alpha Monitor")
root.geometry("300x500")

power_label = tk.Label(root, text="Alpha Power: --")
metric_label = tk.Label(root, text="Metric: --")
hue_label = tk.Label(root, text="Hue Sent: --")
baseline_label = tk.Label(root, text="Baseline Alpha: --")
baseline_display_label = tk.Label(root, text="Baseline Alpha: --")
status_label = tk.Label(root, text="Status: --")

for w in [power_label, metric_label, hue_label, baseline_label, baseline_display_label, status_label]:
    w.pack(pady=5)

tk.Button(root, text="Start Baseline", command=start_baseline).pack(pady=5)

channel_vars = []
for i in range(4):
    var = tk.IntVar(value=1)
    tk.Checkbutton(root, text=f"Include Channel {i+1}", variable=var).pack(anchor="w")
    channel_vars.append(var)

def update_channels():
    global included_channels
    included_channels = [i for i, var in enumerate(channel_vars) if var.get() == 1]
    print(f"[CONFIG] Included channels: {included_channels}")

tk.Button(root, text="Apply Channel Selection", command=update_channels).pack(pady=5)

meter_canvas = tk.Canvas(root, width=100, height=100)
meter_canvas.pack(pady=5)
meter_canvas.create_rectangle(10, 0, 90, 100, outline="black")

# ---------------------------- START ---------------------------- #
if __name__ == "__main__":
    update_channels()
    poll_latest_file()
    root.mainloop()
