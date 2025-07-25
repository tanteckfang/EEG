# 🧠 EEG to Philips Hue Light Control System

This project demonstrates how to use **EEG alpha wave activity** (particularly during relaxation or meditation) to control **Philips Hue smart lights in real-time**. When the user is relaxed (e.g., eyes closed), the lights turn blue. When focus or stress increases (i.e., alpha waves drop), the lights transition toward red.

The system includes:
- GUI controls
- Channel selection
- Real-time EEG decoding
- Alpha smoothing
- Baseline calibration

---

## 🌟 Features

- ✅ Real-time EEG Alpha Power Detection (8–13 Hz)
- ✅ Smoothing with Exponential Moving Average (EMA)
- ✅ Real-time Philips Hue Light Control
- ✅ 4 Customizable Color Sequences
- ✅ Baseline Calibration (12-sec eyes-closed window)
- ✅ Tkinter GUI for Status Display & Channel Control

---

## 🧠 How It Works

1. EEG data is continuously monitored via files generated by OpenBCI GUI.
2. A **12-second baseline** alpha power is recorded while the user keeps their eyes closed.
3. **Alpha power** is calculated using Welch's method and smoothed with EMA.
4. A **normalized metric (0 to 1)** is computed by dividing smoothed alpha by the baseline.
5. Lights respond in real time via Philips Hue bridge using color gradients.

---

## 🛠 Setup Instructions

### 1. Clone This Repository

```bash
git clone https://github.com/tanteckfang/EEG.git
cd EEG
```

---

### 2. Install Python Dependencies

```bash
pip install numpy pandas requests watchdog scipy tkinter nbclient nbformat nest_asyncio tensorflow
```

---

### 3. Set Up OpenBCI

- Launch the **OpenBCI GUI** and start recording EEG.
- Note down the **recording folder name** (e.g., `OpenBCISession_2025-06-11_22-42-47`).
- Make sure `.txt` files are being continuously updated.

---

### 4. Setup Philips Hue (PC Integration)

#### 🔌 Step 1: Connect Hue Bridge to Network

- Connect the Hue Bridge via LAN cable to your router.
- Use the **Philips Hue app** in your phone to initialize the bridge.
- Connect the lights to the Hue Bridge using the app
- Visit :https://www.philips-hue.com/en-my/support/connect-hue-product/bulbs-and-lamps#bridge for more details
    

#### 🌐 Step 2: Find Bridge IP

Visit: [https://discovery.meethue.com/](https://discovery.meethue.com/)  
Copy the `internalipaddress`, e.g., `192.168.1.100`
Or find it using your Philips Hue app at settings -> My Hue system -> i

#### 🔐 Step 3: Get Authorized API Username

1. Visit `http://<bridge_ip>/debug/clip.html`
2. Use:
   - **URL:** `/api`
   - **Method:** `POST`
   - **Body:** `{ "devicetype": "my_eeg_app" }`
3. Press the **physical button on the Hue bridge**
4. Click "POST"
5. Copy the returned `"username"` value.

Update these in your code:
```python
USERNAME = "your_authorized_username"
BRIDGE_IP = "your_bridge_ip"
```

---

### 5. Run the EEG Control System

1. Open `control.py` and update:
   ```python
   base_folder_path = r"C:\Users\<your_username>\Documents\OpenBCI_GUI\Recordings"
   folder = "<your_recording_folder_name>"
   ```
2. In your terminal:
   ```bash
   python control.py
   ```
3. Click **Start Baseline** in the GUI and **close your eyes for ~12 seconds**.
4. After calibration, the system starts real-time monitoring and light control.
5. Lights should reflect your **relaxed vs. focused** mental state.

---

## 🌈 Color Sequences

You can customize the lighting response using:
```python
COLOR_SEQUENCES = {
    "Sequence 1": (0, 46920),     # Red to Blue
    "Sequence 2": (25500, 12750), # Green to Yellow
    "Sequence 3": (46920, 56100), # Blue to Purple
    "Sequence 4": (0, 56100)      # Red to Purple
}
```

---

## 🖥 GUI Overview

| Element                | Description                                  |
|------------------------|----------------------------------------------|
| Alpha Power            | Real-time power in 8–13 Hz range             |
| Metric                 | Value from 0 to 1 (mapped to color hue)      |
| Hue Sent               | The color hue value sent to Philips Hue      |
| Baseline Alpha Power   | Median-based alpha baseline × 0.75           |
| Status                 | Indicates if alpha is below baseline         |
| Channel Checkboxes     | Select which EEG channels to include         |
| Vertical Color Bar     | Green → Yellow → Red based on metric value   |

---

## ⚙ Developer Notes

- **Smoothing** is implemented using Exponential Moving Average:
  ```python
  alpha_smooth = (1 - factor) * alpha_smooth + factor * current_value
  ```
- **Baseline** is calculated as:
  ```python
  baseline = 0.75 × median(alpha_power during calibration)
  ```
- The GUI is built with `Tkinter` for simplicity.
- The Hue color is computed by **linear interpolation** between start and end hues of selected color sequence.

---

## 📁 Directory Structure

```
EEG/
├── control.py                # Main controller script
├── eeg_decode_teckfang.ipynb # Preprocessing notebook (runs automatically)
├── README.md                 # You are here
├── requirements.txt          # Optional: all dependencies
```

---

## 🧪 Example Output

```
[EEG] Alpha Power: 2.29e+01, Smoothed: 2.10e+01, Normalized Metric: 0.84, Hue: 39452
[HUE] Sent hue 39452 to Light 1. Response: 200
[HUE] Sent hue 39452 to Light 2. Response: 200
```

---

## 🙋 Support

If you encounter any issues, feel free to open an issue on this GitHub repo or contact Teck Fang.
