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


