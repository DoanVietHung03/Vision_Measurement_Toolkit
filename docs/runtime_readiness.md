# Checklist Sẵn Sàng Chạy Runtime

Tài liệu này tổng hợp trạng thái hiện tại sau khi gộp và refactor hai pipeline Distance + Velocity/Speed vào package chung `vision_measurement`.

## Kết Luận Nhanh

Source code hiện đã ổn ở mức package, unit test và smoke test không phụ thuộc artifact lớn.

Để chạy thật hai module chính, phần còn thiếu chủ yếu là:

- dependency runtime chưa cài đủ trong environment hiện tại;
- model YOLO/Depth, video đầu vào và file ground-truth chưa có trong `assets/`;
- calibration thật cho từng camera/video chưa được tạo hoặc chưa được điền vào config.

## Trạng Thái Source Hiện Tại

Đã kiểm tra:

```powershell
python -m unittest discover -s tests -v
```

Kết quả:

```text
Ran 19 tests
OK
```

Smoke test đã chạy được:

```powershell
python apps\run_distance_homography.py --config configs\distance_homography.json
python apps\run_distance_depth.py --config configs\distance_depth.json --no-display
python apps\run_speed_lane.py --config configs\speed_vehicle_lane.json --dry-run
```

Ý nghĩa kết quả:

- Homography distance points mode chạy được và trả `distance_m = 3.0`.
- Depth raw-depth smoke chạy được và trả `distance_m = 1.24425`.
- Speed dry-run đọc config được và in ra path/model/calibration readiness.

## Dependency Runtime Còn Thiếu

Environment hiện tại đã có nhiều package chính, nhưng còn thiếu:

```text
supervision
transformers
onnxruntime
```

Ghi chú:

- `supervision` cần cho module speed video UI, polygon zone, annotation và frame generator.
- `transformers` cần cho Depth-Anything inference.
- `onnxruntime` cần cho model ONNX CPU runtime.
- Máy hiện tại báo `torch.cuda.is_available() == False`, nên nếu chạy trên máy này thì nên ưu tiên `onnxruntime` CPU thay vì `onnxruntime-gpu`, trừ khi đã cấu hình lại CUDA đầy đủ.

Lệnh cài gợi ý:

```powershell
python -m pip install supervision transformers onnxruntime
```

Nếu dùng môi trường CUDA/GPU thật:

```powershell
python -m pip install onnxruntime-gpu
```

## Artifact Chung Đang Thiếu

Theo các config hiện tại, các path sau chưa tồn tại:

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

Path đã tồn tại và đang được dùng cho smoke test:

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

Đã chạy được:

- mode `points`;
- mode `image` nếu chỉ dùng ảnh có sẵn và không bật detect/height/undistort;
- logic `PlaneCalibration`, transform point, distance và CSV serialization.

Còn thiếu để chạy đầy đủ workflow cũ:

```text
assets/sample_videos/cam_2.mp4
assets/weights/yolo11n-pose.onnx
configs/generated_camera_calibration.json
```

Còn cần calibration thật:

- `source_points`: 4 điểm trên mặt phẳng trong ảnh/video thật;
- `target_width`, `target_height`: kích thước bird-eye-view;
- `meters_per_pixel`: tỷ lệ mét/pixel sau khi transform;
- camera calibration JSON nếu dùng `--undistort` hoặc `--height`.

Lệnh smoke hiện tại:

```powershell
python apps\run_distance_homography.py --config configs\distance_homography.json
```

Lệnh chạy image không mở UI:

```powershell
python apps\run_distance_homography.py --config configs\distance_homography.json --mode image --no-display
```

Lệnh chạy video sau khi có artifact:

```powershell
python apps\run_distance_homography.py --config configs\distance_homography.json --mode video --no-display --max-frames 100
```

Nếu cần detect person/pose:

```powershell
python apps\run_distance_homography.py --config configs\distance_homography.json --mode video --detect --no-display --max-frames 100
```

Nếu cần height/undistort:

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

Đã chạy được:

- raw-depth smoke mode;
- `MetricDepthProjector`;
- công thức `Z = SCALE_FACTOR / raw_depth`;
- distance 3D từ hai pixel có raw depth.

Còn thiếu để chạy inference/video/person calibration:

```text
transformers
Depth-Anything model cache/download
assets/sample_videos/cam_2.mp4
assets/weights/yolo11n.onnx
```

Còn cần calibration thật:

- `scale_factor` theo scene/camera thật;
- `fx`, `fy`, `cx`, `cy` theo camera thật;
- `reference_width` đúng với resolution calibration;
- điểm pixel `point_a`, `point_b` phù hợp với ảnh/video thật.

Lệnh smoke hiện tại:

```powershell
python apps\run_distance_depth.py --config configs\distance_depth.json --no-display
```

Lệnh one-shot depth inference sau khi có dependency/model:

```powershell
python apps\run_distance_depth.py --config configs\distance_depth.json --infer-depth --no-display
```

Lệnh realtime depth video sau khi có artifact:

```powershell
python apps\run_distance_depth.py --config configs\distance_depth.json --realtime-video --no-display --max-frames 100
```

Lệnh calibrate scale bằng người tham chiếu:

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

Đã chạy được:

- dry-run đọc config;
- BEV transform;
- `SpeedEstimator`;
- validation/calibration unit test.

Còn thiếu để chạy video tracking thật:

```text
supervision
onnxruntime
assets/sample_videos/road.mp4
assets/weights/yolo12n.onnx
```

Còn thiếu nếu cần validate/calibrate bằng ground truth:

```text
assets/ground_truth/gt_data.pkl
```

Optional:

```text
assets/sample_images/lane_mask.png
```

Còn cần calibration thật cho đúng video/camera:

- `source_points`: 4 điểm polygon lane trên frame;
- `target_width`, `target_height`: kích thước BEV;
- `meters_per_pixel`: tỷ lệ thực tế;
- `target_classes`: class id YOLO đúng với model đang dùng.

Lệnh dry-run hiện tại:

```powershell
python apps\run_speed_lane.py --config configs\speed_vehicle_lane.json --dry-run
```

Lệnh video smoke sau khi có artifact:

```powershell
python apps\run_speed_lane.py --config configs\speed_vehicle_lane.json --no-display --max-frames 100
```

Lệnh validate PKL sau khi có ground truth:

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

Output mong đợi:

```text
configs/generated_camera_calibration.json
```

Ảnh checkerboard hiện có:

```text
assets/checkerBoard/calibration_images/*.jpg
```

Lệnh tạo calibration JSON:

```powershell
python apps\run_camera_calibration.py --config configs\camera_calibration.json --no-undistorted
```

Cần kiểm tra lại:

- `pattern_size` hiện là `[9, 6]`;
- `square_size_m` hiện là `0.034`;
- nếu detect được quá ít ảnh checkerboard thì cần đổi pattern size, square size hoặc dùng bộ ảnh calibration khác.

## Thứ Tự Làm Để Chạy Thật

1. Cài dependency runtime còn thiếu:

```powershell
python -m pip install supervision transformers onnxruntime
```

2. Đặt model vào:

```text
assets/weights/yolo11n-pose.onnx
assets/weights/yolo11n.onnx
assets/weights/yolo12n.onnx
```

Hoặc sửa config để trỏ tới model thật đang có.

3. Đặt video vào:

```text
assets/sample_videos/cam_2.mp4
assets/sample_videos/road.mp4
```

Hoặc sửa config để trỏ tới video thật đang có.

4. Tạo camera calibration:

```powershell
python apps\run_camera_calibration.py --config configs\camera_calibration.json --no-undistorted
```

5. Chỉnh calibration cho từng module:

- `configs/distance_homography.json`
- `configs/distance_depth.json`
- `configs/speed_vehicle_lane.json`

6. Chạy smoke có giới hạn frame:

```powershell
python apps\run_distance_homography.py --config configs\distance_homography.json --mode video --no-display --max-frames 100
python apps\run_distance_depth.py --config configs\distance_depth.json --realtime-video --no-display --max-frames 100
python apps\run_speed_lane.py --config configs\speed_vehicle_lane.json --no-display --max-frames 100
```

7. Nếu smoke pass, mở UI để kiểm tra annotation:

```powershell
python apps\run_distance_homography.py --config configs\distance_homography.json --mode video --detect
python apps\run_distance_depth.py --config configs\distance_depth.json --realtime-video
python apps\run_speed_lane.py --config configs\speed_vehicle_lane.json
```

## Acceptance Gate Để Xem Là Sẵn Sàng Chạy

Module Distance được xem là sẵn sàng khi:

- có video/ảnh đúng scene;
- có YOLO pose model nếu cần detect/height;
- có camera calibration nếu cần undistort/height;
- homography source points và meters-per-pixel đã được calibrate;
- depth scale factor và intrinsics đã được calibrate;
- chạy được `--max-frames 100` không lỗi.

Module Speed được xem là sẵn sàng khi:

- có video road thật;
- có YOLO detect model;
- có `supervision` và ONNX runtime;
- lane source points và meters-per-pixel đúng với video;
- chạy được `--max-frames 100` và có speed label hợp lý;
- nếu có `gt_data.pkl`, validation sai số nằm trong ngưỡng chấp nhận.

## Ghi Chú Về Git

Theo yêu cầu hiện tại, không cần và không nên thực hiện bất kỳ thao tác Git nào trong quá trình bổ sung artifact, chạy smoke hoặc chỉnh config.
