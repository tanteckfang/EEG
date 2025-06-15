# EEG to Philips Hue Light Control System

This project demonstrates how to use EEG alpha wave activity (particularly during relaxation or meditation) to control Philips Hue smart lights in real-time. When the user is relaxed (e.g., eyes closed), the lights turn blue. When stress or focus increases (alpha waves drop), the lights transition toward red. The system includes GUI controls and smoothing for a more stable visual effect.

---

## 🌟 Features

- **EEG Alpha Power Detection** (8–13 Hz)
- **Real-time Light Control** via Philips Hue
- **Customizable Color Sequences**
- **EEG Smoothing Mechanism**
- **GUI with Status Display & Channel Selection**
- **Baseline Calibration via Eye Closure**

---

## 🧠 How It Works

- The system monitors EEG data recorded by OpenBCI GUI.
- A **10–12 second baseline** is recorded while the user keeps eyes closed.
- The **alpha power** is computed using Welch’s method, smoothed, then compared against the baseline.
- A **normalized metric (0–1)** is computed.
- Philips Hue lights are updated using a color mapping from red (low alpha) to blue (high alpha).

---

## 🛠 Setup Instructions

### 1. Clone This Repo

```bash
git clone https://github.com/tanteckfang/EEG.git
cd EEG

### 2. Install Python Dependencies
pip install numpy pandas requests watchdog scipy tkinter nbclient nbformat nest_asyncio tensorflow

### 3. Set Up OpenBCI
Use OpenBCI GUI to start recording EEG data.
record the name of the file
start the recording

### 4. Philips Hue Setup (PC Integration)
Step 1: Connect Hue Bridge to WiFi
Plug in the Philips Hue Bridge.

Connect to your router (via LAN cable).

Use the Philips Hue app to initialize the bridge.

Step 2: Find Your Bridge IP
Visit https://discovery.meethue.com/

Note the internalipaddress, e.g., 192.168.1.100

Step 3: Create an Authorized API Username
Open a browser and run:
http://<bridge_ip>/debug/clip.html
In the page:

URL: /api

Method: POST

Body: { "devicetype": "my_eeg_app" }

Press the physical button on the Hue bridge.

Click "POST". If successful, it returns a username like:

"username": "ZqKauJLZ3x4je5RxtKnaTAK958AIYRBYmMHNqhEw"
Update this in your code:
USERNAME = "your_authorized_username"
BRIDGE_IP = "your_bridge_ip"

### 5. How to Run
1. change the base_folder_path and folder in the main function in control.py according to your file

2. At your terminal run python control.py

3. Press Start Baseline and close your eyes for ~12 seconds.

4. Once baseline is recorded, live alpha monitoring begins.

5. Watch your selected lights respond to your mental state.

6.Adjust the baseline until your desired baseline which is able to turn to 1 if you are relaxed.

### 6. Color Sequences
In the code, you can select different color sequences:
COLOR_SEQUENCES = {
    "Sequence 1": (0, 46920),     # Red to Blue
    "Sequence 2": (25500, 12750), # Green to Yellow
    "Sequence 3": (46920, 56100), # Blue to Purple
    "Sequence 4": (0, 56100)      # Red to Purple
}

### 7. GUI Overview
Alpha Power: Real-time power in 8–13Hz band.

Metric: Normalized from 0 to 1 (used to control light).

Hue Sent: Actual color value sent to Hue lights.

Baseline Alpha Power: Displayed once you run baseline.

Status: Shows whether alpha is below baseline.

Channel Checkboxes: Select EEG channels to include.

Bar Meter: Visual indicator from green to red.

###  Developer Notes
EEG smoothing uses an Exponential Moving Average (EMA), can change the smoothing factor yourself.

The baseline is calculated as 75% of the median alpha power during calibration.

Lights use a linear interpolation between hue values.


