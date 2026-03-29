Hãy đóng vai một Senior Prompt Engineer kiêm Senior Machine Learning Engineer chuyên xây dựng code machine learning production-ready, clean code, dễ bảo trì, dễ mở rộng.

Tôi sẽ cung cấp cho bạn 3 đầu vào:
1. Mẫu dataset đã được tiền xử lý (định dạng CSV) được lưu tại: `dataset\test_train_scaled`
2. File README mô tả quá trình tiền xử lý dữ liệu: `doc/DATA_PREPROCESSING_HANDOFF.md`
3. File mô tả metric đánh giá: `doc/METRIC.md`

Nhiệm vụ của bạn:
Phân tích toàn bộ các đầu vào trên và sinh ra cho tôi một chương trình Python hoàn chỉnh dùng XGBoost để dự đoán thời tiết.

Bối cảnh và ràng buộc:
- Bài toán này là regression.
- Mọi lệnh chạy code (cài thư viện, train, infer, evaluate, export) bắt buộc phải thực thi trong môi trường ảo của project (ưu tiên `.venv`).
- Dataset đầu vào đã là dữ liệu sau tiền xử lý, nhưng bạn vẫn phải kiểm tra và xử lý an toàn các vấn đề còn lại nếu có, ví dụ: kiểu dữ liệu, missing values, datetime, feature selection.
- Chỉ được sử dụng thông tin xuất hiện trong dataset mẫu, file README tiền xử lý và file metric.
- Không được tự bịa thêm cột dữ liệu, không được tự suy diễn target ngoài phạm vi hợp lý từ đầu vào.
- Nếu README hoặc metric chưa đủ rõ, bạn được phép đưa ra giả định hợp lý, nhưng bắt buộc phải liệt kê rõ ràng từng giả định trước khi viết code.

Mục tiêu đầu ra:
Tạo một pipeline đầy đủ, sạch và có thể dùng làm nền cho project thật, bao gồm:
- đọc dữ liệu đã tiền xử lý
- kiểm tra dữ liệu đầu vào
- tách feature/target
- chia train/validation/test
- huấn luyện XGBoost Regressor
- dự đoán
- đánh giá mô hình
- lưu model
- load model
- suy luận cho dữ liệu mới
- sinh biểu đồ đánh giá
- xuất file Excel kết quả
- lưu toàn bộ artifact vào thư mục output theo timestamp runtime

Yêu cầu bắt buộc về thiết kế code:
- Code phải sạch, rõ ràng, dễ đọc, dễ bảo trì.
- Không viết kiểu notebook.
- Không viết kiểu demo sơ sài.
- Chỉ viết Python script chuẩn, có thể copy-paste chạy được.
- Tư duy như đang xây cho một ML project thật.
- Ưu tiên tính đúng đắn, tổ chức code tốt và tính mở rộng.

Yêu cầu kỹ thuật bắt buộc:
1. Dùng `XGBRegressor`.
2. Tách hàm rõ ràng, tối thiểu phải có:
   - `load_data`
   - `preprocess_data`
   - `split_data`
   - `build_model`
   - `train_model`
   - `predict`
   - `evaluate_model`
   - `save_model`
   - `load_model`
   - `run_training_pipeline`
   - `run_inference` hoặc `predict_new_samples`
3. Nếu có cột datetime, phải xử lý rõ ràng và an toàn.
4. Nếu có missing values, phải xử lý rõ ràng.
5. Nếu có categorical features, phải xử lý rõ ràng.
6. Chỉ scaling hoặc encoding nếu thực sự hợp lý; nếu không cần thì nói rõ là không cần.
7. Thêm `type hints`.
8. Thêm `docstring` ngắn gọn cho các hàm chính.
9. Có `if __name__ == "__main__":`
10. Có xử lý lỗi cơ bản bằng `try/except` ở các đoạn quan trọng.
11. Không hardcode bừa bãi; phải gom cấu hình vào một nơi, ví dụ class config hoặc dataclass config:
   - data path
   - target column
   - random seed
   - test size
   - validation size
   - model hyperparameters
   - output directory
12. Tên biến, tên hàm, tên file phải chuyên nghiệp, nhất quán.
13. Không import thừa.
14. Không để hàm chết, code thừa, logic lặp lại vô ích.

Yêu cầu về đánh giá mô hình:
- Vì đây là regression, phải hỗ trợ và in rõ các metric phù hợp như:
   - MAE
   - MSE
   - RMSE
   - R2
- Các metric phải được format dễ đọc, không in kiểu quá rối.
- Phải tạo biểu đồ đánh giá và lưu ra file, ví dụ các biểu đồ phù hợp như:
   - Actual vs Predicted
   - Residual distribution
   - Feature importance (nếu phù hợp với XGBoost)
- Phải xuất file Excel chứa kết quả dự đoán và metric.
- Toàn bộ output phải được lưu vào thư mục dạng:
  `output/<timestamp_run>/...`

Yêu cầu về output trả lời của bạn:
Hãy trả lời đúng theo cấu trúc sau:

## Phần 1: Phân tích nhanh đầu vào
- Phân tích dataset mẫu
- Phân tích README tiền xử lý
- Phân tích file metric
- Chỉ ra các điểm đã rõ và các điểm còn thiếu thông tin

## Phần 2: Xác định bài toán và giả định
- Xác nhận đây là bài toán regression
- Xác định target column dựa trên đầu vào
- Liệt kê rõ toàn bộ giả định đang dùng trước khi code

## Phần 3: Thiết kế chương trình
- Giải thích ngắn gọn cấu trúc chương trình sẽ tạo
- Nêu rõ các thành phần chính của pipeline

## Phần 4: Code Python hoàn chỉnh
- Xuất ra một file Python hoàn chỉnh
- Code phải đầy đủ, chạy được, không pseudo-code
- Bao gồm train pipeline, evaluate, save/load model, inference, lưu biểu đồ, lưu Excel output

## Phần 5: Hướng dẫn chạy
- Liệt kê các thư viện cần cài và phiên bản vào requirements.txt
- Câu lệnh chạy
- Cấu trúc file/thư mục mong đợi

## Phần 6: Gợi ý cải tiến
- Đề xuất các hướng cải tiến hợp lý để nâng cao chất lượng mô hình trong tương lai
- Không làm phức tạp quá mức nếu đầu vào hiện tại chưa yêu cầu

Ưu tiên thư viện:
- pandas
- numpy
- scikit-learn
- xgboost
- joblib
- matplotlib
- openpyxl hoặc xlsxwriter nếu cần xuất Excel

Lưu ý rất quan trọng:
- Không được bỏ sót phần inference.
- Không được bỏ sót phần evaluate.
- Không được bỏ sót phần export biểu đồ và Excel.
- Không được tự bịa thêm logic ngoài đầu vào.
- Nếu thiếu thông tin, phải nói rõ giả định.
- Mục tiêu cuối cùng là tạo ra một chương trình sạch, chuyên nghiệp, có thể dùng làm nền cho project thật.