# 5-Point Landmark Face Recognition & Tracking System

High-accuracy face recognition and tracking system using 5-point facial landmark alignment (Haar cascades / MediaPipe), ArcFace 512-D ONNX embeddings, and closed-loop servo tracking via ESP8266.

## Camera Configuration

The system automatically detects and prioritizes **physical external USB cameras** (e.g. `Wed Camera`) over the PC's built-in laptop camera (`Integrated Camera`) and virtual cameras (`EShare Virtual Camera`).

### List Connected Cameras
To see all connected cameras and which device is selected:
```bash
python -m src.camera --list-cams
```

Output:
```text
=================================================================
INDEX   | TYPE                 | DEVICE NAME
=================================================================
  0     | Internal PC Camera   | Integrated Camera
  1     | Virtual Camera       | EShare Virtual Camera
  2     | Physical External    | Wed Camera -> [SELECTED PHYSICAL]
=================================================================
```

### Automatic Physical Camera Usage
All entry points automatically select the physical camera by default (`--cam auto`):
- **Camera Test / Live HUD**: `python -m src.camera`
- **Face Detection**: `python -m src.detect`
- **Target Face Enrollment**: `python -m src.enroll --name TargetName`
- **Target Tracking & Servo Base**: `python -m src.track_target --target TargetName --port COM13`

### Manual Camera Override
If you want to manually specify a camera index or network stream:
```bash
# Explicit camera index (e.g. 2)
python -m src.track_target --target TargetName --cam 2

# IP / RTSP / HTTP Camera Stream
python -m src.track_target --target TargetName --cam http://192.168.1.100:8080/video
```

## Wireless Tachometer Stepper Tracking (MQTT + NodeMCU)

Tracks an enrolled face horizontally across the frame and maps its position to a tachometer needle (0° to 180°):
- **Model / Vision Client**: Uses ArcFace identity lock-in and 5-point landmark detection.
- **MQTT Protocol**: Transmits raw angle degrees (`0` to `180`) via `paho-mqtt` ("peho") to Eclipse Mosquitto broker.
- **Microcontroller**: NodeMCU ESP8266 subscribes over WiFi to `tachometer/angle` and drives a stepper motor needle via AccelStepper.

### Run Tracker
```bash
python -m src.mqtt_tachometer_tracker --target Sonia
```
*(Options: `--broker localhost`, `--topic tachometer/angle`, `--axis vertical|horizontal`, `--invert`, `--offset 0.0`)*

### Hotkeys
- `C` : Zero-calibrate needle to current head position
- `[` / `]` : Nudge zero offset by -5° / +5°
- `V` : Switch tracking axis (Vertical / Horizontal)
- `I` : Invert direction
- `0` : Home needle to 0°
- `R` : Center needle to 90°

### Firmware
Firmware file is located at [`firmware/nodemcu_stepper_tachometer/nodemcu_stepper_tachometer.ino`](file:///c:/Users/user/Videos/face-recognition-5pt/firmware/nodemcu_stepper_tachometer/nodemcu_stepper_tachometer.ino).

## Whole-Person Tracking & Face-Association (Honest 3-State Model)

Maintains temporal identity continuity across temporary facial loss through an **honest three-state model** with **exponential confidence decay**:
- **CONFIRMED**: Person body is tracked and actively verified via ArcFace face embedding ($C \approx 0.70 - 1.0$).
- **TEMPORARY**: Face is temporarily lost (e.g., head turn, looking away, momentary occlusion), but the whole-person body track continues to be observed via Kalman/Pose tracking. Confidence decays exponentially:
  $$C(t) = C_0 \cdot \exp(-\lambda \Delta t) \quad \text{where} \quad \lambda = \frac{\ln 2}{t_{1/2}}$$
  If the face reappears on this body before confidence falls below threshold, identity restores to **CONFIRMED** with zero identity switch.
- **UNKNOWN**: Confidence has decayed below the confirmation threshold ($C < 0.35$), or the track never possessed a verified face.

### Closed-Loop Target Centering Logic (Virtual PTZ Gimbal)
Operates completely without physical hardware components:
- Closed-loop software controller computes normalized pixel error $(e_x, e_y)$ relative to optical frame center.
- Features deadband filtering ($r_{\text{dead}} = 0.06$) to eliminate jitter when target is centered.
- Implements simulated 2-axis Pan-Tilt gimbal dynamics and renders a dynamic, zoomed Picture-in-Picture virtual camera viewport that smoothly tracks the target in software.

### Run Live Webcam Tracker
```bash
python -m src.person_face_tracker --target Sonia
```
*(Options: `--cam auto` or `--cam 0`, `--decay-half-life 2.5`, `--match-thresh 0.40`)*

**Interactive Hotkeys:**
- `H` : Toggle simulated head turn (suppresses facial landmarks to test Temporary decay state)
- `O` : Toggle simulated obstacle occluder (tests walking behind barriers)
- `P` : Toggle Virtual PTZ zoomed software gimbal window
- `T` : Cycle selected target track
- `C` : Recenter virtual gimbal to 0°
- `Q` / `ESC` : Quit session

### Run Automated Scenario Evaluation Suite
Evaluates Head Turns, Obstacle Occlusions, and Two-Person Path Crossings:
```bash
python -m src.evaluate_scenarios --target Sonia
```
Generates quantitative metrics in [`data/evaluation_metrics.json`](file:///c:/Users/user/Videos/face-recognition-5pt/data/evaluation_metrics.json) and publication-quality diagnostic plots in [`evaluation_results.png`](file:///c:/Users/user/Videos/face-recognition-5pt/evaluation_results.png).


