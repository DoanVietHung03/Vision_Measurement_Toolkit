# Runtime Readiness Checklist

Tai lieu nay tong hop trang thai hien tai sau khi gop va refactor hai pipeline
Distance + Velocity/Speed vao package chung `vision_measurement`.

## Ket Luan Nhanh

Source code hien da on o muc package, unit test va smoke test khong phu thuoc
artifact lon.

De chay that hai module chinh, phan con thieu chu yeu la:

- dependency runtime chua cai du trong environment hien tai;
- model YOLO/Depth, video dau vao va file ground-truth chua co trong `assets/`;
- calibration that cho tung camera/video chua duoc tao hoac chua duoc dien vao config.

## Trang Thai Source Hien Tai

Da kiem tra:

```powershell
python -m unittest discover -s tests -v
```

Ket qua:

```text
Ran 19 tests
OK
```

Smoke test da chay duoc:

```powershell
python apps\run_distance_homography.py --config configs\distance_homography.json
python apps\run_distance_depth.py --config configs\distance_depth.json --no-display
python apps\run_speed_lane.py --config configs\speed_vehicle_lane.json --dry-run
```

Ket qua y nghia:

- Homography distance points mode chay duoc va tra `distance_m = 3.0`.
- Depth raw-depth smoke chay duoc va tra `distance_m = 1.24425`.
- Speed dry-run doc config duoc va in ra path/model/calibration readiness.

## Dependency Runtime Con Thieu

Environment hien tai da co nhieu package chinh, nhung con thieu:

```text
supervision
transformers
onnxruntime
```

Ghi chu:

- `supervision` can cho module speed video UI, polygon zone, annotation va frame generator.
- `transformers` can cho Depth-Anything inference.
- `onnxruntime` can cho model ONNX CPU runtime.
- May hien tai bao `torch.cuda.is_available() == False`, nen neu chay tren may nay thi nen uu tien `onnxruntime` CPU thay vi `onnxruntime-gpu`, tru khi da cau hinh lai CUDA day du.

Lenh cai goi goi y:

```powershell
python -m pip install supervision transformers onnxruntime
```

Neu dung moi truong CUDA/GPU that:

```powershell
python -m pip install onnxruntime-gpu
```

## Artifact Chung Dang Thieu

Theo cac config hien tai, cac path sau chua ton tai:

```text
assets/sample_videos/cam_2.mp4
assets/sample_videos/road.mp4
configs/generated_camera_calibration.json
assets/weights/yolo11n-pose.onnx
assets/weights/yolo11n.onnx
assets/weights/yolo12n.onnx
assets/ground_truth/gt_data.pkl
assets/sample_images/lane_mask.png
```

Path da ton tai va dang duoc dung cho smoke test:

```text
assets/checkerBoard/calibration_images/calib_000.jpg
```

## Module Distance

### 1. Homography Distance

App:

```text
apps/run_distance_homography.py
```

Config:

```text
configs/distance_homography.json
```

Da chay duoc:

- mode `points`;
- mode `image` neu chi dung anh co san va khong bat detect/height/undistort;
- logic `PlaneCalibration`, transform point, distance, CSV serialization.

Con thieu de chay day du workflow cu:

```text
assets/sample_videos/cam_2.mp4
assets/weights/yolo11n-pose.onnx
configs/generated_camera_calibration.json
```

Con can calibration that:

- `source_points`: 4 diem tren mat phang trong anh/video that;
- `target_width`, `target_height`: kich thuoc bird-eye-view;
- `meters_per_pixel`: ty le met/pixel sau khi transform;
- camera calibration JSON neu dung `--undistort` hoac `--height`.

Lenh smoke hien tai:

```powershell
python apps\run_distance_homography.py --config configs\distance_homography.json
```

Lenh chay image khong mo UI:

```powershell
python apps\run_distance_homography.py --config configs\distance_homography.json --mode image --no-display
```

Lenh chay video sau khi co artifact:

```powershell
python apps\run_distance_homography.py --config configs\distance_homography.json --mode video --no-display --max-frames 100
```

Neu can detect person/pose:

```powershell
python apps\run_distance_homography.py --config configs\distance_homography.json --mode video --detect --no-display --max-frames 100
```

Neu can height/undistort:

```powershell
python apps\run_distance_homography.py --config configs\distance_homography.json --mode video --detect --height --undistort --no-display --max-frames 100
```

### 2. Depth Distance

App:

```text
apps/run_distance_depth.py
```

Config:

```text
configs/distance_depth.json
```

Da chay duoc:

- raw-depth smoke mode;
- `MetricDepthProjector`;
- cong thuc `Z = SCALE_FACTOR / raw_depth`;
- distance 3D tu hai pixel co raw depth.

Con thieu de chay inference/video/person calibration:

```text
transformers
Depth-Anything model cache/download
assets/sample_videos/cam_2.mp4
assets/weights/yolo11n.onnx
```

Con can calibration that:

- `scale_factor` theo scene/camera that;
- `fx`, `fy`, `cx`, `cy` theo camera that;
- `reference_width` dung voi resolution calibration;
- diem pixel `point_a`, `point_b` phu hop voi anh/video that.

Lenh smoke hien tai:

```powershell
python apps\run_distance_depth.py --config configs\distance_depth.json --no-display
```

Lenh one-shot depth inference sau khi co dependency/model:

```powershell
python apps\run_distance_depth.py --config configs\distance_depth.json --infer-depth --no-display
```

Lenh realtime depth video sau khi co artifact:

```powershell
python apps\run_distance_depth.py --config configs\distance_depth.json --realtime-video --no-display --max-frames 100
```

Lenh calibrate scale bang nguoi tham chieu:

```powershell
python apps\run_distance_depth.py --config configs\distance_depth.json --infer-depth --calibrate-person --real-distance-m 15.3 --no-display
```

## Module Speed / Velocity

App:

```text
apps/run_speed_lane.py
```

Config:

```text
configs/speed_vehicle_lane.json
```

Da chay duoc:

- dry-run doc config;
- BEV transform;
- `SpeedEstimator`;
- validation/calibration unit test.

Con thieu de chay video tracking that:

```text
supervision
onnxruntime
assets/sample_videos/road.mp4
assets/weights/yolo12n.onnx
```

Con thieu neu can validate/calibrate bang ground truth:

```text
assets/ground_truth/gt_data.pkl
```

Optional:

```text
assets/sample_images/lane_mask.png
```

Con can calibration that cho dung video/camera:

- `source_points`: 4 diem polygon lane tren frame;
- `target_width`, `target_height`: kich thuoc BEV;
- `meters_per_pixel`: ty le thuc te;
- `target_classes`: class id YOLO dung voi model dang dung.

Lenh dry-run hien tai:

```powershell
python apps\run_speed_lane.py --config configs\speed_vehicle_lane.json --dry-run
```

Lenh video smoke sau khi co artifact:

```powershell
python apps\run_speed_lane.py --config configs\speed_vehicle_lane.json --no-display --max-frames 100
```

Lenh validate PKL sau khi co ground truth:

```powershell
python apps\run_speed_lane.py --config configs\speed_vehicle_lane.json --validate-pkl assets/ground_truth/gt_data.pkl
```

## Camera Calibration

App:

```text
apps/run_camera_calibration.py
```

Config:

```text
configs/camera_calibration.json
```

Output mong doi:

```text
configs/generated_camera_calibration.json
```

Anh checkerboard hien co:

```text
assets/checkerBoard/calibration_images/*.jpg
```

Lenh tao calibration JSON:

```powershell
python apps\run_camera_calibration.py --config configs\camera_calibration.json --no-undistorted
```

Can kiem tra lai:

- `pattern_size` hien la `[9, 6]`;
- `square_size_m` hien la `0.034`;
- neu detect duoc qua it anh checkerboard thi can doi pattern size, square size hoac bo anh calibration khac.

## Thu Tu Lam De Chay That

1. Cai dependency runtime con thieu:

```powershell
python -m pip install supervision transformers onnxruntime
```

2. Dat model vao:

```text
assets/weights/yolo11n-pose.onnx
assets/weights/yolo11n.onnx
assets/weights/yolo12n.onnx
```

Hoac sua config de tro toi model that dang co.

3. Dat video vao:

```text
assets/sample_videos/cam_2.mp4
assets/sample_videos/road.mp4
```

Hoac sua config de tro toi video that dang co.

4. Tao camera calibration:

```powershell
python apps\run_camera_calibration.py --config configs\camera_calibration.json --no-undistorted
```

5. Chinh calibration cho tung module:

- `configs/distance_homography.json`
- `configs/distance_depth.json`
- `configs/speed_vehicle_lane.json`

6. Chay smoke co gioi han frame:

```powershell
python apps\run_distance_homography.py --config configs\distance_homography.json --mode video --no-display --max-frames 100
python apps\run_distance_depth.py --config configs\distance_depth.json --realtime-video --no-display --max-frames 100
python apps\run_speed_lane.py --config configs\speed_vehicle_lane.json --no-display --max-frames 100
```

7. Neu smoke pass, moi mo UI de kiem tra annotation:

```powershell
python apps\run_distance_homography.py --config configs\distance_homography.json --mode video --detect
python apps\run_distance_depth.py --config configs\distance_depth.json --realtime-video
python apps\run_speed_lane.py --config configs\speed_vehicle_lane.json
```

## Acceptance Gate De Xem La San Sang Chay

Module Distance duoc xem la san sang khi:

- co video/anh dung scene;
- co YOLO pose model neu can detect/height;
- co camera calibration neu can undistort/height;
- homography source points va meters-per-pixel da duoc calibrate;
- depth scale factor va intrinsics da duoc calibrate;
- chay duoc `--max-frames 100` khong loi.

Module Speed duoc xem la san sang khi:

- co video road that;
- co YOLO detect model;
- co `supervision` va ONNX runtime;
- lane source points va meters-per-pixel dung voi video;
- chay duoc `--max-frames 100` va co speed label hop ly;
- neu co `gt_data.pkl`, validation sai so nam trong nguong chap nhan.

## Ghi Chu Ve Git

Theo yeu cau hien tai, khong can va khong nen thuc hien bat ky thao tac Git nao
trong qua trinh bo sung artifact, chay smoke hay chinh config.
