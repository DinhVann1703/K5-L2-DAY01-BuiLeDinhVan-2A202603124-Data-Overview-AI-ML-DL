# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468; `class_name`="cab" ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`="ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Record này nói về **cả bức ảnh** vì bài toán phân loại ảnh (Classification) chỉ đoán 1 nhãn đại diện chung cho toàn bộ bức ảnh, không chỉ ra vị trí hay đóng khung riêng từng vật thể.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- Đi kèm taxonomy/checkpoint (bộ nhãn ImageNet-1K đi cùng model yolo11n-cls.pt).
- Model không tự nghĩ ra được tên lớp mà chỉ chọn từ danh sách 1.000 nhãn đã được dạy từ trước.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- Để tránh nhầm lẫn: Tên lớp cab có thể xuất hiện ở nhiều bộ dữ liệu khác nhau với mã số (class_id) khác nhau.
- Để máy và người cùng hiểu: Máy tính cần mã số (class_id), người đọc cần tên chữ (class_name), và taxonomy_name cho biết danh sách này lấy từ bộ chuẩn nào.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Guideline cần đưa ra quy tắc ưu tiên:

    - Chọn vật to nhất, nằm ở chính giữa hoặc rõ nét nhất ảnh.

    - Nếu có nhiều vật ngang nhau, quy định rõ thứ tự ưu tiên (ví dụ: ưu tiên chọn người trước phương tiện xe cộ).
6. Vì sao không được lấy `score` của model làm ground truth?
- score chỉ là mức độ "tự tin" của model khi đoán (ở đây đoán đúng/sai gì cũng ra khoảng 0.51), chứ không chắc chắn là kết quả đúng ngoài đời thật.
- Model vẫn có thể đoán sai, còn ground truth phải là nhãn do con người kiểm tra và xác nhận là đúng 100%.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- Góc trên-trái : Nằm tại tọa độ ngang pixel 385.33 và tọa độ dọc pixel 69.24 (tính từ mép trên cùng của ảnh xuống).
- Góc dưới-phải : Nằm tại tọa độ ngang pixel 498.92 và tọa độ dọc pixel 348.92. 
- Kích thước pixel: Khung hộp có chiều rộng là 113.58 pixel và chiều cao là 279.68 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- Ngưỡng 0.20: có 17 prediction (vật thể).
- Ngưỡng 0.60: có 6 prediction (vật thể).
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- Lọc bỏ hộp sai (False Positives): Phải soi kỹ và bấm xóa rất nhiều hộp rác, hộp dự đoán nhầm nền gạch/bức tường thành vật thể.
- Xử lý trùng lặp (Duplicate Boxes): Phải xóa các hộp bị vẽ đè lên nhau cùng chỉ một vật thể do ngưỡng thấp khiến bộ lọc NMS giữ lại nhiều hộp thừa.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- Quy tắc bao khít (Tight Bounding Box): "Khung hộp phải vẽ sát khít 4 mép ngoài cùng của vật thể, không để chừa khoảng trống thừa quá 2–3 pixel và tuyệt đối không cắt cụt bất kỳ bộ phận nhìn thấy nào của vật thể."
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- Vật thể bị che : Chiếc lò/bếp bên trái (oven 0.69) bị chiếc bàn gỗ phía trước che khuất một nửa phần thân dưới; hoặc cánh tay người bên góc trái (person 0.61) bị cắt mép sát viền ảnh.
- Viết thành quy tắc trong Guideline vì tình trạng vật thể bị che khuất (occlusion) hoặc chạm viền mép ảnh (truncation) xảy ra liên tục ở hầu hết mọi bức ảnh thực tế. Guideline cần quy định rõ từ đầu (ví dụ: "Nếu vật bị che khuất trên 80% thì bỏ qua, nếu bị che dưới 80% thì chỉ vẽ khung bao quanh phần nhìn thấy được").
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- Đường biên thực tế chi tiết từng pixel: Bounding box chỉ là một hình chữ nhật bao quanh nên chứa rất nhiều điểm ảnh nền thừa; trong khi polygon uốn lượn ôm sát theo đúng đường cong cơ thể, nếp áo, bờ vai và hai ống quần của người.
- Diện tích và cấu trúc hình học chuẩn xác: Cho biết diện tích bề mặt thực tế, các góc gấp khúc và phần rỗng/khe hở giữa các bộ phận (như khoảng trống giữa hai chân hoặc giữa cánh tay và thân mình).
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- Công dụng: Dùng để phân biệt các cá thể riêng biệt trong cùng một bức ảnh. 
- class_id là mã định danh chung cho cả loài/lớp.
- tracking_id dùng trong video để theo dõi một vật di chuyển xuyên suốt qua nhiều khung hình theo thời gian; còn instance_id chỉ có giá trị nội bộ, độc lập trên duy nhất bức ảnh tĩnh hiện tại.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- Quy tắc bám sát đường biên thực tế (Tight Pixel Boundary Rule): "Điểm polygon phải được chấm khít theo đường viền nhìn thấy ngoài cùng của vật thể, sai số lệch pixel cho phép tối đa 1–2 pixel; không được cắt lẹm vào thân vật thể và không được bao trùm nền xung quanh."
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- Vị trí cụ thể: Mặt trên của chiếc bàn ăn (dining table) có các vật dụng đặt đè lên (như chiếc khuôn bánh bowl 0.74, âu bột và hũ gia vị), làm đường biên của mặt bàn bị che khuất và chạm sát đáy các vật này.
- Chọn Guidenline vì trường hợp vật thể nằm đè lên nhau hoặc che khuất nhau là việc xảy ra liên tục trong các cảnh sinh hoạt đời thực. Guideline cần quy định trước một quy tắc chuẩn thống nhất (ví dụ: "Khi vật nhỏ đặt đè lên bàn, polygon của bàn phải đi vòng qua mép ngoài của các vật đặt trên nó, hoặc polygon của từng vật vẽ riêng biệt theo ranh giới nhìn thấy"). Có sẵn quy tắc trong guideline sẽ giúp annotator thao tác tự động,
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn duy nhất (tên lớp) cho cả ảnh. | Ảnh traffic có nhiều đối tượng (xe buýt, ô tô, người) nhưng model chỉ đoán 1 nhãn bao trùm là cab với score = 0.51 (khó chọn vật thể đại diện). |Nhìn toàn bộ ảnh, đối chiếu quy tắc ưu tiên trong guideline để chọn đúng 1 nhãn chính xác nhất.  |Kiểm tra nhãn được gán có đúng theo quy tắc ưu tiên của guideline hay không; sửa lại nếu annotator chọn sai chủ thể.  |
| Phát hiện vật thể |Tập hợp các khung hộp tọa độ hình chữ nhật kèm tên lớp (bbox_xyxy + class_name) . | Trong ảnh kitchen, lò nướng bên trái (oven 0.69) bị mặt bàn che khuất một phần; hạ ngưỡng xuống 0.20 sinh ra nhiều hộp rác/trùng lặp (17 vật thể). | Kéo chuột vẽ khung chữ nhật ôm khít lấy từng vật thể nhìn thấy được và chọn đúng tên lớp. |Soi 4 cạnh hộp xem có bị vẽ lẹm hay quá rộng không; xóa các hộp rác/trùng lặp hoặc vẽ bổ sung các vật bị bỏ sót.  |
| Instance segmentation | Tập hợp các chuỗi điểm đa giác viền khép kín (polygon_xy) kèm instance_id và class_name. | Trong ảnh kitchen, các vật thể trên bàn ăn (dining table 0.58) đè lên nhau khiến đường biên mask của bàn bị thủng hoặc dính liền vào các bát/khuôn bánh (bowl). |Chấm từng điểm ảnh viền theo sát mép ngoài của từng vật thể riêng biệt; tách riêng từng thực thể dù chạm sát nhau.  |Soát độ chính xác của đường biên mask xem có bám sát biên dạng thực tế không; kiểm tra các điểm giao cắt giữa 2 vật chạm nhau.  |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Tuyệt đối không đưa họ tên, MSSV, email, số điện thoại hoặc bất kỳ thông tin cá nhân/dữ liệu nhạy cảm nào vào nội dung báo cáo và các file output.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Dừng thực hiện ngay lập tức và báo cáo trực tiếp cho Lab Coach / Giảng viên hướng dẫn (thực hiện quy trình escalation).

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
