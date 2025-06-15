# EEG Decode and Light Control - Teck Fang's Project (Notebook-Driven Integration)

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
nest_asyncio.apply()

# ---------- CONFIG ---------- #
BRIDGE_IP = "192.168.226.220"
USERNAME = "ZqKauJLZ3x4je5RxtKnaTAK958AIYRBYmMHNqhEw"
LIGHT_IDS = {"Light 1": "1", "Light 2": "2"}
COLOR_SEQUENCES = {
    "Sequence 1": (0, 46920),     # Red to Blue
    "Sequence 2": (25500, 12750), # Green to Yellow
    "Sequence 3": (46920, 56100), # Blue to Purple
    "Sequence 4": (0, 56100)      # Red to Purple
}
SELECTED_SEQUENCE = "Sequence 1"
SELECTED_LIGHTS = ["Light 1", "Light 2"]
FS = 250
WINDOW_LEN_S = 6  # Increased from 4 to 6 seconds
AMP_THRESHOLD = 80e-6  # 80 microvolts threshold for artifact rejection
last_valid_metric = 0.5

# ---------- HUE CONTROL ---------- #
def send_to_light(hue_val):
    payload = {
        "on": True,
        "bri": 200,
        "hue": int(hue_val),
        "sat": 254
    }
    for name in SELECTED_LIGHTS:
        light_id = LIGHT_IDS.get(name)
        if light_id:
            url = f"http://{BRIDGE_IP}/api/{USERNAME}/lights/{light_id}/state"
            try:
                response = requests.put(url, json=payload, timeout=1)
                print(f"[HUE] Sent hue {int(hue_val)} to {name}. Response: {response.status_code}")
            except Exception as e:
                print(f"[HUE] Failed to send to light {name}: {e}")

# ---------- EEG PROCESSING ---------- #
def is_clean(eeg):
    max_amp = np.max(np.abs(eeg))
    print(f"[ARTIFACT CHECK] Max Amplitude: {max_amp:.2e}")
    if max_amp > AMP_THRESHOLD:
        print(f"[CLEAN CHECK] Rejected due to high amplitude: {max_amp:.2e} > {AMP_THRESHOLD}")
        return False
    return True

def process_eeg_chunk(eeg_data):
    global last_valid_metric
    eeg_data = eeg_data[:, :4]
    if not is_clean(eeg_data):
        metric = last_valid_metric
        alpha_power = 0.0
        print("[EEG] Noisy data detected. Using last valid metric.")
    else:
        freqs, psd = welch(eeg_data.mean(axis=1), fs=FS, nperseg=FS)
        alpha_band = (freqs >= 8) & (freqs <= 13)
        alpha_power = np.trapz(psd[alpha_band], freqs[alpha_band])

        # Normalize alpha power to [0, 1] with more practical scale adjustment
        alpha_min, alpha_max = 0.0, 25.0  # Adjusted normalization scale
        metric = np.clip((alpha_power - alpha_min) / (alpha_max - alpha_min), 0, 1)

        last_valid_metric = metric

    hue_start, hue_end = COLOR_SEQUENCES[SELECTED_SEQUENCE]
    hue_val = (1 - metric) * hue_start + metric * hue_end

    print(f"[EEG] Alpha Power: {alpha_power:.2e}, Normalized Metric: {metric:.2f}, Hue: {int(hue_val)}")
    send_to_light(hue_val)
    update_gui(alpha_power, metric, hue_val)

# ---------- FILE MONITORING ---------- #
class FileHandler(FileSystemEventHandler):
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

# ---------- GUI DISPLAY ---------- #
def update_gui(power, metric, hue_val, status=""):
    power_label.config(text=f"Alpha Power: {power:.2e}")
    metric_label.config(text=f"Metric: {metric:.2f}")
    hue_label.config(text=f"Hue Sent: {int(hue_val)}")
    meter_canvas.delete("meter")
    bar_height = int(metric * 100)
    color = "#00ff00" if metric < 0.33 else "#ffff00" if metric < 0.66 else "#ff0000"
    meter_canvas.create_rectangle(10, 100 - bar_height, 90, 100, fill=color, tags="meter")
    root.update_idletasks()

# ---------- NOTEBOOK EXECUTION ---------- #
def run_notebook():
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

    thread = threading.Thread(target=task, daemon=True)
    thread.start()

# ---------- MAIN ---------- #
def main():
    run_notebook()
    print("Monitoring EEG files for updates...")
    base_folder_path = r"C:\\Users\\ttfta\\Documents\\OpenBCI_GUI\\Recordings"
    folder = "OpenBCISession_2025-06-11_23-14-22"
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

# ---------- INIT GUI ---------- #
root = tk.Tk()
root.title("EEG Alpha Monitor")
root.geometry("300x260")
power_label = tk.Label(root, text="Alpha Power: --")
power_label.pack(pady=5)
metric_label = tk.Label(root, text="Metric: --")
metric_label.pack(pady=5)
hue_label = tk.Label(root, text="Hue Sent: --")
hue_label.pack(pady=5)
status_label = tk.Label(root, text="Status: --")
status_label.pack(pady=5)
meter_canvas = tk.Canvas(root, width=100, height=100)
meter_canvas.pack(pady=5)
meter_canvas.create_rectangle(10, 0, 90, 100, outline="black")

if __name__ == "__main__":
    main()
