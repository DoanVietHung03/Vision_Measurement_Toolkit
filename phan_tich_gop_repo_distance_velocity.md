# Phân tích việc gộp hai repo `Distance_calculation` và `Velocity_detection`

## 1. Kết luận ngắn gọn

Nên gộp hai repo `Distance_calculation` và `Velocity_detection` vào một repo chung, nhưng **không nên gộp theo kiểu copy nguyên hai project rồi để cạnh nhau**. Cách hợp lý nhất là tái cấu trúc thành một repo lớn hơn theo hướng **Computer Vision Measurement Toolkit**, trong đó:

- `calibration` là module dùng chung.
- `distance` là module xử lý đo khoảng cách.
- `speed` là module xử lý đo vận tốc.
- `apps` chứa các script chạy demo hoặc pipeline hoàn chỉnh.
- `experiments` chứa code thử nghiệm, benchmark, validate.
- `legacy` chứa code cũ ban đầu để đối chiếu trong quá trình refactor.

Lý do chính là: bài toán đo tốc độ trong `Velocity_detection` thực chất phụ thuộc mạnh vào các bước đo khoảng cách, calibration, Bird’s Eye View và quy đổi pixel sang mét. Đây cũng là các vấn đề mà repo `Distance_calculation` đang tập trung xử lý. Vì vậy, nếu tổ chức lại đúng cách, việc gộp sẽ giúp project sạch hơn, dễ mở rộng hơn và chuyên nghiệp hơn.

---

## 2. Hiện trạng hai repo

### 2.1. Repo `Distance_calculation`

Repo này đang tập trung vào bài toán **đo khoảng cách thực tế trên ảnh 2D**. Theo README, project thử nghiệm nhiều phương pháp khác nhau, bao gồm:

- Homography.
- Stereo Vision.
- Deep Learning.

Mục tiêu là đánh giá từng phương pháp dựa trên:

- Độ chính xác.
- Tốc độ xử lý.
- Khả năng áp dụng trong các tình huống thực tế.

Cấu trúc hiện tại có các thư mục và file đáng chú ý:

```text
Distance_calculation/
├── DL_method/
│   ├── distance_cal.py
│   ├── instruction.md
│   └── scale_factor_cal.py
│
├── Homography_method/
│   ├── calibrate.py
│   ├── height_estimator.py
│   ├── instruction.md
│   ├── main.py
│   ├── stabilizer.py
│   └── video.py
│
├── checkerBoard/
│   ├── calibration_images/
│   └── create_imgs.py
│
├── Visualize_Comparison.py
├── export_onnx.py
├── frame_extractor.py
├── metric.md
└── requirements.txt
```

Nhận xét:

- Repo này đang giống một **research/experiment repo** hơn là một package hoàn chỉnh.
- Có nhiều hướng thử nghiệm khác nhau, nhưng phần calibration, scale factor, video handling và đánh giá nên được chuẩn hóa lại.
- Các script như `main.py`, `calibrate.py`, `distance_cal.py`, `scale_factor_cal.py` có thể được tách thành module rõ ràng hơn.

---

### 2.2. Repo `Velocity_detection`

Repo này tập trung vào bài toán **đo tốc độ phương tiện**, đặc biệt trong thư mục `Vehicle_Lane`.

Theo README, project có hai hướng test:

1. Có file `*.pkl` và ảnh `lane_mask`.
2. Không có file `*.pkl` và `lane_mask`, khi đó cần biết kích thước thật của đường/làn đường để tự cấu hình Bird’s Eye View.

Các bước chính trong README gồm:

- Chạy `calibration_tool.py` để lấy các thông tin như `SOURCE_POINTS`, `TARGET_WIDTH`, `TARGET_HEIGHT`, `METERS_PER_PIXEL`.
- Chạy `main.py` để test.
- Nếu không có dữ liệu hỗ trợ, cần xác định 4 điểm trên mặt đường/làn đường, cấu hình kích thước target và tính `METERS_PER_PIXEL` bằng công thức:

```text
METERS_PER_PIXEL = real_size_of_the_road / number_of_pixels_in_TARGET_WIDTH
```

Cấu trúc đáng chú ý:

```text
Velocity_detection/
└── Vehicle_Lane/
    ├── calibration_tool.py
    ├── config.py
    ├── export_onnx.py
    ├── main.py
    ├── speed_estimator.py
    ├── transformer.py
    └── validate_accuracy.py
```

Nhận xét:

- Repo này có pipeline rõ hơn cho bài toán velocity.
- Tuy nhiên, nó đang chứa lại các phần đáng ra có thể dùng chung với repo đo khoảng cách: calibration, Bird’s Eye View, scale factor, transform, validation.
- Nếu để riêng lâu dài, phần calibration và quy đổi pixel/mét rất dễ bị lặp lại hoặc bị lệch logic giữa hai repo.

---

## 3. Vì sao nên gộp hai repo?

### 3.1. Vì hai repo cùng thuộc một domain

Cả hai repo đều xử lý bài toán **đo đại lượng thực tế từ ảnh hoặc video**.

- `Distance_calculation`: đo khoảng cách, chiều cao, scale factor, homography.
- `Velocity_detection`: đo vận tốc dựa trên vị trí vật thể qua thời gian, Bird’s Eye View và mét/pixel.

Về bản chất, đo tốc độ là bài toán mở rộng từ đo khoảng cách:

```text
speed = real_world_distance / time
```

Muốn đo được tốc độ, hệ thống phải giải quyết được ít nhất ba vấn đề:

1. Xác định vị trí vật thể trong ảnh/video.
2. Chuyển vị trí ảnh sang không gian tương ứng với thực tế.
3. Tính quãng đường thực tế theo thời gian.

Như vậy, repo `Velocity_detection` phụ thuộc rất nhiều vào các thành phần mà `Distance_calculation` đang nghiên cứu.

---

### 3.2. Tái sử dụng được calibration

Đây là lý do quan trọng nhất.

Trong cả hai repo đều có những khái niệm giống nhau hoặc liên quan trực tiếp:

- Homography.
- Camera calibration.
- Bird’s Eye View.
- Source points.
- Target points.
- Pixel-to-meter ratio.
- Scale factor.
- Real-world coordinate estimation.

Nếu để riêng, mỗi repo sẽ dễ có một cách tính riêng. Điều này dẫn đến các vấn đề:

- Kết quả distance và velocity có thể không đồng nhất.
- Sửa bug calibration ở repo này nhưng quên sửa ở repo kia.
- Khi đổi cách tính `METERS_PER_PIXEL`, phải sửa nhiều nơi.
- Khó kiểm soát sai số vì pipeline đo lường bị phân mảnh.

Khi gộp lại, nên đưa toàn bộ phần này vào module dùng chung:

```text
src/vision_measurement/calibration/
├── homography.py
├── bird_eye_view.py
├── scale_factor.py
├── camera_calibration.py
└── perspective_transform.py
```

Khi đó cả module `distance` và `speed` đều dùng chung một nguồn logic.

---

### 3.3. Giảm trùng lặp code

Hai repo hiện đều có các loại script như:

- `main.py`.
- `export_onnx.py`.
- file config.
- script calibration.
- script validate/evaluate.
- xử lý video/frame.

Nếu giữ riêng, project sẽ dễ bị trùng:

```text
Distance_calculation/Homography_method/main.py
Velocity_detection/Vehicle_Lane/main.py

Distance_calculation/export_onnx.py
Velocity_detection/Vehicle_Lane/export_onnx.py

Distance_calculation/Homography_method/video.py
Velocity_detection/Vehicle_Lane/main.py hoặc transformer.py
```

Sau khi gộp, nên phân biệt rõ:

- Module lõi nằm trong `src/`.
- Script chạy demo nằm trong `apps/`.
- Script thử nghiệm nằm trong `experiments/`.
- Config nằm trong `configs/`.

Như vậy code không bị rải rác, và người dùng project sẽ hiểu ngay nên chạy file nào.

---

### 3.4. Dễ mở rộng thành một hệ thống đo lường hoàn chỉnh

Nếu chỉ giữ repo `Distance_calculation`, project bị giới hạn ở đo khoảng cách.

Nếu chỉ giữ repo `Velocity_detection`, project bị giới hạn ở đo tốc độ.

Nhưng nếu gộp lại, project có thể phát triển thành một toolkit lớn hơn:

- Đo khoảng cách giữa hai điểm.
- Đo chiều cao vật thể.
- Đo tốc độ phương tiện.
- Ước lượng vị trí vật thể trong mặt phẳng thực tế.
- Đánh giá sai số đo lường.
- So sánh Homography, Deep Learning, Stereo Vision.
- Tích hợp object detection/tracking.

Tên repo khi đó có thể đổi thành:

```text
Vision_Measurement
Distance_Speed_Estimation
RealWorld_Measurement_CV
Camera_Measurement_Toolkit
Traffic_Vision_Measurement
```

Nếu mục tiêu là project cá nhân/portfolio, tên nên rõ ràng và dễ hiểu nhất:

```text
Distance_Speed_Estimation
```

Nếu mục tiêu là mở rộng lâu dài, tên nên rộng hơn:

```text
Vision_Measurement
```

---

### 3.5. Dễ viết tài liệu và demo hơn

Nếu hai repo tách riêng, README của mỗi repo chỉ giải thích được một phần. Người đọc sẽ không thấy được liên hệ giữa:

```text
calibration -> distance estimation -> speed estimation
```

Khi gộp lại, README chính có thể trình bày pipeline tổng quát:

```text
Input image/video
    ↓
Camera calibration / Homography / Bird’s Eye View
    ↓
Pixel-to-meter conversion
    ↓
Distance estimation
    ↓
Speed estimation over time
    ↓
Evaluation / Visualization
```

Điều này làm project dễ hiểu hơn rất nhiều, đặc biệt nếu dùng để đưa vào CV, portfolio hoặc báo cáo đồ án.

---

## 4. Rủi ro nếu gộp sai cách

Việc gộp là nên làm, nhưng nếu gộp không có cấu trúc thì project sẽ rối hơn.

### 4.1. Rối do nhiều file `main.py`

Nếu copy nguyên hai repo vào một repo chung:

```text
NewRepo/
├── Distance_calculation/
│   └── Homography_method/main.py
└── Velocity_detection/
    └── Vehicle_Lane/main.py
```

thì người dùng sẽ không biết nên chạy file nào trước, file nào là bản mới, file nào là thử nghiệm.

Cách xử lý:

- Không để nhiều `main.py` ở các vị trí mơ hồ.
- Đổi thành các entrypoint rõ nghĩa:

```text
apps/run_distance_demo.py
apps/run_speed_demo.py
apps/run_calibration_tool.py
apps/run_evaluation.py
```

---

### 4.2. Rối do lẫn code thử nghiệm và code lõi

Repo `Distance_calculation` hiện có nhiều phần mang tính thử nghiệm. Điều này không xấu, nhưng nếu trộn trực tiếp với pipeline chạy chính thì sẽ khó maintain.

Cách xử lý:

- Code ổn định đưa vào `src/`.
- Code thử nghiệm đưa vào `experiments/`.
- Code cũ chưa refactor đưa vào `legacy/`.

Ví dụ:

```text
legacy/
├── distance_calculation_old/
└── velocity_detection_old/

experiments/
├── compare_distance_methods/
├── homography_experiments/
└── speed_validation/
```

---

### 4.3. Rối do config bị phân tán

Hiện `Velocity_detection` có `config.py`, trong khi `Distance_calculation` có nhiều script tự xử lý thông số riêng.

Nếu gộp lại, nên chuyển dần sang config dạng `.yaml` hoặc `.json` để tách thông số khỏi code.

Ví dụ:

```text
configs/
├── distance_homography.yaml
├── distance_dl.yaml
├── speed_vehicle_lane.yaml
└── camera_calibration.yaml
```

Mục tiêu:

- Dễ thay video/input.
- Dễ thay thông số calibration.
- Dễ chạy nhiều experiment.
- Không cần sửa code mỗi lần đổi `SOURCE_POINTS`, `TARGET_WIDTH`, `TARGET_HEIGHT`, `METERS_PER_PIXEL`.

---

### 4.4. Rối do đặt tên module chưa thống nhất

Các tên như `transformer.py`, `calibrate.py`, `distance_cal.py`, `scale_factor_cal.py` có thể hiểu được trong repo nhỏ, nhưng khi gộp vào repo lớn thì nên đặt rõ hơn.

Ví dụ đổi tên:

```text
calibrate.py              -> camera_calibration.py hoặc homography_calibrator.py
transformer.py            -> perspective_transform.py hoặc bird_eye_transformer.py
distance_cal.py           -> distance_estimator.py
scale_factor_cal.py       -> scale_factor_estimator.py
speed_estimator.py        -> giữ nguyên, vì tên đã rõ
validate_accuracy.py      -> speed_validation.py hoặc evaluate_speed.py
```

---

## 5. Cấu trúc repo đề xuất

Cấu trúc hợp lý nhất nên là:

```text
Vision_Measurement/
│
├── README.md
├── requirements.txt
├── pyproject.toml                 # Có thể thêm sau nếu muốn chuẩn hóa package
├── .gitignore
│
├── configs/
│   ├── distance_homography.yaml
│   ├── distance_dl.yaml
│   ├── speed_vehicle_lane.yaml
│   └── camera_calibration.yaml
│
├── src/
│   └── vision_measurement/
│       │
│       ├── calibration/
│       │   ├── __init__.py
│       │   ├── camera_calibration.py
│       │   ├── homography.py
│       │   ├── bird_eye_view.py
│       │   ├── perspective_transform.py
│       │   └── scale_factor.py
│       │
│       ├── distance/
│       │   ├── __init__.py
│       │   ├── homography_distance.py
│       │   ├── dl_distance.py
│       │   ├── stereo_distance.py
│       │   └── height_estimator.py
│       │
│       ├── speed/
│       │   ├── __init__.py
│       │   ├── speed_estimator.py
│       │   ├── trajectory.py
│       │   └── vehicle_speed_pipeline.py
│       │
│       ├── detection/
│       │   ├── __init__.py
│       │   ├── detector.py
│       │   └── onnx_inference.py
│       │
│       ├── tracking/
│       │   ├── __init__.py
│       │   └── tracker.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── video.py
│           ├── frame_extractor.py
│           ├── visualization.py
│           ├── geometry.py
│           └── metrics.py
│
├── apps/
│   ├── run_distance_demo.py
│   ├── run_speed_demo.py
│   ├── run_calibration_tool.py
│   └── run_evaluation.py
│
├── experiments/
│   ├── distance_method_comparison/
│   ├── homography_experiments/
│   ├── dl_distance_experiments/
│   └── speed_validation/
│
├── docs/
│   ├── overview.md
│   ├── calibration.md
│   ├── distance_estimation.md
│   ├── speed_estimation.md
│   └── migration_notes.md
│
├── tests/
│   ├── test_homography.py
│   ├── test_scale_factor.py
│   ├── test_distance_estimator.py
│   └── test_speed_estimator.py
│
├── legacy/
│   ├── distance_calculation_old/
│   └── velocity_detection_old/
│
└── assets/
    ├── sample_images/
    ├── sample_videos/
    └── calibration_samples/
```

---

## 6. Vai trò của từng phần sau khi gộp

### 6.1. `src/vision_measurement/calibration/`

Đây là phần quan trọng nhất sau khi gộp.

Nên chứa toàn bộ logic liên quan đến:

- Homography matrix.
- Perspective transform.
- Bird’s Eye View.
- Camera calibration bằng checkerboard.
- Tính scale factor.
- Tính `METERS_PER_PIXEL`.
- Chuyển đổi tọa độ ảnh sang tọa độ thực tế.

Module này sẽ được dùng bởi cả `distance` và `speed`.

---

### 6.2. `src/vision_measurement/distance/`

Chứa các phương pháp đo khoảng cách:

- Homography-based distance.
- Deep Learning-based distance.
- Stereo Vision-based distance.
- Height estimation nếu vẫn muốn giữ.

Mục tiêu của module này là trả ra các giá trị thực tế như:

```text
distance_meters
height_meters
real_world_position
```

---

### 6.3. `src/vision_measurement/speed/`

Chứa logic đo tốc độ:

- Lưu trajectory của object qua nhiều frame.
- Tính quãng đường đi được trong mặt phẳng thực tế.
- Tính tốc độ theo FPS hoặc timestamp.
- Smooth kết quả tốc độ nếu cần.
- Validate kết quả với ground truth nếu có.

Module này không nên tự tính lại homography hoặc scale factor. Nó nên gọi lại module `calibration`.

---

### 6.4. `src/vision_measurement/detection/`

Chứa wrapper cho object detection model.

Ví dụ:

- Load ONNX model.
- Chạy inference.
- Chuẩn hóa output bbox.
- Trả về object position cho module distance/speed.

Việc có module này sẽ giúp project dễ mở rộng sang YOLOX, RT-DETR hoặc các detector khác.

---

### 6.5. `apps/`

Chứa các file chạy chính cho người dùng.

Không nên để người dùng phải mò trong `src/` để chạy. Các lệnh nên rõ ràng:

```bash
python apps/run_calibration_tool.py --config configs/camera_calibration.yaml
python apps/run_distance_demo.py --config configs/distance_homography.yaml
python apps/run_speed_demo.py --config configs/speed_vehicle_lane.yaml
```

---

### 6.6. `experiments/`

Chứa các thử nghiệm chưa chắc ổn định:

- So sánh Homography vs Deep Learning.
- Test nhiều cách tính scale factor.
- Validate sai số speed.
- Vẽ chart so sánh.
- Benchmark FPS.

Các file như `Visualize_Comparison.py`, `metric.md`, các thử nghiệm trong `DL_method`, `Homography_method` có thể chuyển dần vào đây.

---

### 6.7. `legacy/`

Dùng để chứa code gốc ban đầu sau khi gộp bằng `git subtree`.

Không nên xóa code cũ ngay. Nên giữ lại trong giai đoạn đầu để:

- Đối chiếu kết quả.
- Tránh mất logic đang chạy được.
- Refactor từng phần một cách an toàn.

Sau khi đã refactor ổn định, có thể giữ `legacy/` trong vài tuần hoặc vài version rồi mới xóa.

---

## 7. Cách gộp repo an toàn

### 7.1. Không nên copy thủ công ngay từ đầu

Copy thủ công dễ mất lịch sử commit và khó truy vết thay đổi.

Nên dùng `git subtree` để import hai repo vào repo mới.

Ví dụ:

```bash
mkdir Vision_Measurement
cd Vision_Measurement
git init

# Add remote của repo đo khoảng cách
git remote add distance https://github.com/DoanVietHung03/Distance_calculation.git

# Add remote của repo đo tốc độ
git remote add velocity https://github.com/DoanVietHung03/Velocity_detection.git

# Fetch code từ hai repo
git fetch distance
git fetch velocity

# Import vào thư mục legacy
git subtree add --prefix=legacy/distance_calculation_old distance master
git subtree add --prefix=legacy/velocity_detection_old velocity master
```

Sau bước này, chưa refactor vội. Hãy commit trạng thái import trước.

```bash
git add .
git commit -m "Import distance and velocity repositories into legacy folders"
```

---

### 7.2. Refactor theo từng phase

Không nên refactor toàn bộ một lần. Nên chia thành các phase nhỏ.

#### Phase 1: Import nguyên trạng

Mục tiêu:

- Giữ code cũ chạy được.
- Không thay đổi logic.
- Chỉ đưa hai repo vào `legacy/`.

Kết quả mong muốn:

```text
legacy/distance_calculation_old/
legacy/velocity_detection_old/
```

---

#### Phase 2: Tách module calibration dùng chung

Mục tiêu:

- Lấy logic từ `calibrate.py`, `calibration_tool.py`, `transformer.py`, `scale_factor_cal.py`.
- Đưa vào `src/vision_measurement/calibration/`.

Kết quả mong muốn:

```text
src/vision_measurement/calibration/
├── homography.py
├── bird_eye_view.py
├── perspective_transform.py
└── scale_factor.py
```

Đây là phase quan trọng nhất vì nó quyết định project sau khi gộp có sạch hay không.

---

#### Phase 3: Tách module distance

Mục tiêu:

- Chuyển logic đo khoảng cách từ `DL_method`, `Homography_method` sang `src/vision_measurement/distance/`.
- Chuẩn hóa input/output.

Ví dụ output nên thống nhất dạng:

```python
{
    "distance_m": 12.4,
    "method": "homography",
    "confidence": None,
    "metadata": {
        "source_points": ..., 
        "meters_per_pixel": ...
    }
}
```

---

#### Phase 4: Tách module speed

Mục tiêu:

- Chuyển `speed_estimator.py`, `validate_accuracy.py` và logic trong `main.py` sang `src/vision_measurement/speed/`.
- Module speed chỉ nên nhận dữ liệu đã quy đổi sang không gian thực tế hoặc gọi module calibration để quy đổi.

Ví dụ flow:

```text
bbox center in image
    ↓
calibration.image_to_world()
    ↓
real-world position
    ↓
speed_estimator.update(object_id, position, timestamp)
    ↓
speed_kmh
```

---

#### Phase 5: Tạo app chạy demo

Mục tiêu:

- Không chạy trực tiếp file trong `src/`.
- Tạo các file chạy rõ nghĩa trong `apps/`.

Ví dụ:

```text
apps/run_distance_demo.py
apps/run_speed_demo.py
apps/run_calibration_tool.py
```

---

#### Phase 6: Chuẩn hóa config

Mục tiêu:

- Chuyển các thông số như `SOURCE_POINTS`, `TARGET_WIDTH`, `TARGET_HEIGHT`, `METERS_PER_PIXEL`, model path, video path sang YAML.

Ví dụ:

```yaml
input:
  video_path: assets/sample_videos/road.mp4

calibration:
  source_points:
    - [100, 300]
    - [500, 300]
    - [900, 700]
    - [50, 700]
  target_width: 800
  target_height: 1200
  meters_per_pixel: 0.05

speed:
  fps: 30
  smoothing_window: 5
  output_unit: kmh
```

---

#### Phase 7: Viết lại README chính

README chính nên có các phần:

```text
1. Project overview
2. Features
3. Repository structure
4. Installation
5. Quick start
6. Calibration workflow
7. Distance estimation workflow
8. Speed estimation workflow
9. Evaluation
10. Roadmap
```

---

## 8. Kiến trúc pipeline đề xuất

### 8.1. Pipeline đo khoảng cách

```text
Image / Video Frame
    ↓
Camera Calibration / Homography
    ↓
Pixel-to-meter conversion
    ↓
Distance Estimator
    ↓
Distance in meters
    ↓
Visualization / Evaluation
```

---

### 8.2. Pipeline đo tốc độ

```text
Video
    ↓
Object Detection
    ↓
Object Tracking
    ↓
Bird’s Eye View / Homography Transform
    ↓
Pixel-to-meter conversion
    ↓
Trajectory in real-world plane
    ↓
Speed Estimation
    ↓
Speed in km/h
    ↓
Visualization / Validation
```

---

### 8.3. Quan hệ giữa các module

```text
calibration
    ├── được dùng bởi distance
    └── được dùng bởi speed

detection
    └── cung cấp bbox/object center cho speed hoặc distance

tracking
    └── cung cấp object trajectory cho speed

utils
    └── dùng chung cho video, visualization, metrics
```

---

## 9. Những thứ nên giữ riêng sau khi gộp

Dù gộp repo, không có nghĩa là mọi thứ phải trộn lại.

Nên giữ riêng:

- Logic đo khoảng cách và logic đo tốc độ.
- Config cho distance và config cho speed.
- Experiment cho từng phương pháp.
- Dataset/sample video/sample image.
- Tài liệu hướng dẫn từng pipeline.

Nói cách khác, nên gộp ở cấp repo và module dùng chung, nhưng vẫn giữ boundary rõ ràng giữa các feature.

---

## 10. Những thứ không nên làm

Không nên làm:

```text
NewRepo/
├── Distance_calculation/
└── Velocity_detection/
```

Cách này chỉ là đặt hai repo cạnh nhau, chưa thật sự gộp.

Không nên để:

```text
NewRepo/
├── main.py
├── main2.py
├── test.py
├── test_speed.py
├── new_main.py
├── config.py
├── config_old.py
└── final_final.py
```

Cách này sẽ khiến project rất khó hiểu sau vài tuần.

Không nên sửa toàn bộ code ngay sau khi import. Nên import vào `legacy/` trước, sau đó refactor từng phần.

---

## 11. Đề xuất tên repo

### Nếu muốn tên ngắn, rõ mục tiêu

```text
Distance_Speed_Estimation
```

Phù hợp nếu project chủ yếu tập trung vào đo khoảng cách và tốc độ.

---

### Nếu muốn tên rộng hơn để mở rộng lâu dài

```text
Vision_Measurement
```

Phù hợp nếu sau này muốn thêm:

- đo kích thước vật thể,
- đo chiều cao,
- đo khoảng cách theo nhiều phương pháp,
- tracking,
- camera calibration toolkit,
- industrial/traffic measurement.

---

### Nếu muốn nhấn mạnh ứng dụng giao thông

```text
Traffic_Vision_Measurement
```

Phù hợp nếu hướng chính là xe cộ, làn đường, tốc độ phương tiện.

---

## 12. Roadmap refactor đề xuất

### Giai đoạn 1: Gộp an toàn

- Tạo repo mới.
- Import hai repo vào `legacy/` bằng `git subtree`.
- Viết README mô tả mục tiêu gộp.
- Chưa sửa logic cũ.

### Giai đoạn 2: Chuẩn hóa calibration

- Tách homography.
- Tách Bird’s Eye View.
- Tách scale factor.
- Tách camera calibration.
- Viết test nhỏ cho các hàm transform.

### Giai đoạn 3: Chuẩn hóa distance

- Tách Homography distance.
- Tách DL distance.
- Tách height estimator.
- Chuẩn hóa output theo mét.

### Giai đoạn 4: Chuẩn hóa speed

- Tách speed estimator.
- Tách trajectory logic.
- Kết nối speed với calibration module.
- Validate lại kết quả speed.

### Giai đoạn 5: Làm sạch repo

- Di chuyển script demo vào `apps/`.
- Di chuyển thử nghiệm vào `experiments/`.
- Di chuyển tài liệu vào `docs/`.
- Xóa hoặc archive code legacy không cần nữa.

---

## 13. Kết luận cuối

Nên gộp `Distance_calculation` và `Velocity_detection` vì hai repo có cùng bản chất: **đo các đại lượng thực tế từ ảnh/video bằng Computer Vision**.

`Velocity_detection` không nên tồn tại như một project hoàn toàn tách biệt, vì để đo tốc độ chính xác thì nó cần các thành phần nền tảng như calibration, homography, Bird’s Eye View và pixel-to-meter conversion. Đây lại chính là những phần mà `Distance_calculation` đang thử nghiệm.

Tuy nhiên, cách gộp tốt nhất không phải là đặt hai repo cạnh nhau. Cách tốt nhất là:

```text
Một repo chung
    ├── calibration dùng chung
    ├── distance riêng
    ├── speed riêng
    ├── detection/tracking nếu cần
    ├── apps để chạy demo
    ├── experiments để thử nghiệm
    └── legacy để giữ code cũ trong giai đoạn refactor
```

Nếu làm theo hướng này, project sẽ không rối hơn mà ngược lại sẽ rõ ràng hơn, dễ maintain hơn và có giá trị hơn nếu dùng làm portfolio hoặc phát triển tiếp thành một hệ thống đo lường hoàn chỉnh.

---

## 14. Nguồn tham khảo

- `Distance_calculation`: https://github.com/DoanVietHung03/Distance_calculation
- `Velocity_detection`: https://github.com/DoanVietHung03/Velocity_detection
- `Velocity_detection/Vehicle_Lane`: https://github.com/DoanVietHung03/Velocity_detection/tree/master/Vehicle_Lane
- `Distance_calculation/Homography_method`: https://github.com/DoanVietHung03/Distance_calculation/tree/master/Homography_method
- `Distance_calculation/DL_method`: https://github.com/DoanVietHung03/Distance_calculation/tree/master/DL_method
- `Distance_calculation/checkerBoard`: https://github.com/DoanVietHung03/Distance_calculation/tree/master/checkerBoard
