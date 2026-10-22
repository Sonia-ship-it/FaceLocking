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

## Whole-Person Tracking & Face-Association (YuNet + YOLOX + ReID)

Modern robotics tracking and face-association system with deep learning, appearance ReID, and robotic automatic locking policies:
- **Face Detection (YuNet)**: Modern deep learning CNN (`cv2.FaceDetectorYN`) running on CPU (<10ms), replacing legacy Haar cascades, with direct 5-point landmark regression.
- **Body Detection (YOLOX-Nano ONNX + MediaPipe Pose)**: Deep learning body bounding box detection across sitting, standing, and partial angles.
- **Appearance Re-Identification (Clothing Color Signature)**: Extracts normalized torso clothing color histograms (ReID) to maintain identity continuity **even when a person turns their back to the camera or covers their face**.
- **Face Recognition (ArcFace)**: 512-D L2-normalized deep embeddings compared via cosine similarity.
- **Honest Three-State Model**: `CONFIRMED` (green), `TEMPORARY` (amber with exponential decay and appearance cushioning), and `UNKNOWN` (crimson security alert).

### Robotic Automatic Locking Policies
The system supports smart facility access and security locking:
1. **Lock by Unknown (`--lock-policy unknown`) [Facility Security Intruder Mode]**:
   - For compounds, facilities, and secure buildings where all authorized occupants are enrolled.
   - When an unverified stranger / intruder appears in the facility, the robotic system **automatically locks onto them** immediately and tracks their movements.
2. **Lock by Known (`--lock-policy known`) [Authorized Follow Mode]**:
   - Follows and centers onto enrolled facility members (or a specified target like `--target Sonia`).
3. **Manual Lock (`--lock-policy manual`)**:
   - Operator-controlled tracking.
   - Press **`M`** at any time during runtime to cycle between policies!

### Closed-Loop Target Centering Logic (Virtual PTZ Gimbal)
Operates completely without physical hardware components:
- Closed-loop software controller computes normalized pixel error $(e_x, e_y)$ relative to optical frame center.
- Features deadband filtering ($r_{\text{dead}} = 0.06$) to eliminate jitter when target is centered.
- Implements simulated 2-axis Pan-Tilt gimbal dynamics and renders a dynamic, zoomed Picture-in-Picture virtual camera viewport that smoothly tracks the target in software.

### Run Live Webcam Tracker
```bash
python -m src.person_face_tracker --lock-policy unknown
```
*(Options: `--cam internal` or `--cam 0`, `--lock-policy known`, `--target Sonia`, `--decay-half-life 2.5`)*

**Interactive Hotkeys:**
- `M` : Toggle Locking Policy (Lock Unknown Intruder <-> Lock Known Member <-> Manual)
- `H` : Toggle simulated head turn (simulates facial loss to test Temporary decay)
- `O` : Toggle simulated obstacle occluder (tests walking behind barriers)
- `P` : Toggle Virtual PTZ zoomed software gimbal window
- `F` : Toggle Fullscreen mode
- `U` : Toggle HUD Overlay on/off (view clean camera feed)
- `T` : Cycle selected target track
- `C` : Recenter virtual gimbal to 0°
- `Q` / `ESC` : Quit session

### Run Automated Scenario Evaluation Suite
Evaluates Head Turns, Obstacle Occlusions, and Two-Person Path Crossings:
```bash
python -m src.evaluate_scenarios --target Sonia
```
Generates quantitative metrics in [`data/evaluation_metrics.json`](file:///c:/Users/user/Videos/face-recognition-5pt/data/evaluation_metrics.json) and publication-quality diagnostic plots in [`evaluation_results.png`](file:///c:/Users/user/Videos/face-recognition-5pt/evaluation_results.png).


