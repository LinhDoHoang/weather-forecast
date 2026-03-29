1. NHÓM CHỈ SỐ CHÍNH XÁC (ACCURACY METRICS)
Nhóm này đo lường sai số về mặt vật lý (độ C, mm, km/h) để biết mô hình "đoán sát" thực tế đến mức nào.
1.1. MAE (Mean Absolute Error - Sai số tuyệt đối trung bình)
Cách tính: Tổng tất cả các trị tuyệt đối của (Giá trị thực tế - Giá trị dự báo), sau đó chia cho tổng số ngày.
Công thức văn bản: MAE = Tổng |Thực tế - Dự báo| / Số ngày.
Ý nghĩa: Đây là sai số trung bình "thật thà" nhất. Nếu MAE = 1.5, người dùng hiểu rằng App thường xuyên lệch khoảng 1.5 đơn vị.
Ví dụ đánh giá: Trong 4 mô hình, mô hình nào có MAE thấp nhất được coi là mô hình có độ ổn định hằng ngày tốt nhất. Với thời tiết Việt Nam, MAE nhiệt độ dưới 2.0 được coi là đạt yêu cầu ứng dụng.
1.2. RMSE (Root Mean Squared Error - Căn bậc hai sai số trung bình bình phương)
Cách tính: Bình phương các sai số -> Tính trung bình các bình phương đó -> Lấy căn bậc hai của kết quả.
Công thức văn bản: RMSE = Căn bậc hai của [ Tổng (Thực tế - Dự báo)^2 / Số ngày ].
Ý nghĩa: RMSE đặc biệt nhạy cảm với các lỗi lớn vì lỗi được "bình phương" lên trước khi tính. Nó cho biết mô hình có hay mắc sai lầm nghiêm trọng không.
Ví dụ đánh giá: Nếu RMSE cao hơn MAE rất nhiều (ví dụ MAE = 1 nhưng RMSE = 4), chứng tỏ mô hình có những ngày sai số cực nặng (sai 7-8 độ). Một mô hình an toàn phải có RMSE gần sát với MAE.
1.3. R-squared (R2 Score - Hệ số xác định)
Cách tính: 1 - (Tổng bình phương sai số của mô hình / Tổng bình phương sai số nếu chỉ lấy giá trị trung bình để đoán).
Công thức văn bản: R2 = 1 - (Sai số mô hình / Biến động tự nhiên của dữ liệu).
Ý nghĩa: Đo mức độ "thông minh" của mô hình. R2 = 1 là hoàn hảo. R2 = 0 là vô dụng.
Ví dụ đánh giá: Nếu R2 = 0.85, mô hình đã học được 85% quy luật thời tiết năm 2021 của Việt Nam. Nếu R2 thấp, mô hình đó không bắt được tính mùa vụ hoặc xu hướng thay đổi thời tiết.

2. NHÓM CHỈ SỐ AN TOÀN (RELIABILITY METRICS)
Nhóm này đánh giá xem mô hình có bị lệch về một phía hay có gây nguy hiểm vào những ngày đặc biệt không.
2.1. MBE (Mean Bias Error - Sai số định kiến trung bình)
Cách tính: Tổng các (Giá trị dự báo - Giá trị thực tế) và chia cho số ngày (không lấy trị tuyệt đối).
Công thức văn bản: MBE = Tổng (Dự báo - Thực tế) / Số ngày.
Ý nghĩa: Cho biết mô hình có xu hướng dự báo "quá cao" (MBE dương) hay "quá thấp" (MBE âm).
Ví dụ đánh giá: Nếu MBE = +0.8, App của bạn luôn có xu hướng dự báo nóng hơn thực tế. Bạn có thể dùng số này để "trừ bớt" kết quả dự báo trước khi hiển thị cho người dùng để đạt độ chính xác cao hơn.
2.2. Max Error (Sai số cực đại)
Cách tính: Tìm giá trị chênh lệch lớn nhất giữa thực tế và dự báo trong toàn bộ 365 ngày của năm 2021.
Công thức văn bản: Max Error = Giá trị lớn nhất của |Thực tế - Dự báo|.
Ý nghĩa: Tìm ra "điểm yếu nhất" của mô hình.
Ví dụ đánh giá: Nếu Max Error của mô hình RF là 12 độ C vào một ngày bão, mô hình này không đủ an toàn để dùng làm cảnh báo thiên tai, vì nó đã bỏ lỡ một biến động cực kỳ quan trọng.

3. NHÓM CHỈ SỐ ỨNG DỤNG (CLASSIFICATION METRICS)



Đây là phần quan trọng nhất để đưa ra nhắc nhở (Ví dụ: Ngưỡng nắng gắt là 35 độ C).
3.1. Precision (Độ chính xác cảnh báo)
Cách tính: (Số lần báo Nắng gắt ĐÚNG) chia cho (Tổng số lần App đã báo Nắng gắt).
Công thức văn bản: Precision = Số lần đúng / (Số lần đúng + Số lần báo nhầm).
Ý nghĩa: Tránh báo động giả. Người dùng sẽ rất bực mình nếu App liên tục nhắc "Trời sắp mưa" nhưng thực tế trời lại tạnh ráo. Precision cao nghĩa là lời nhắc của App rất uy tín.
3.2. Recall (Độ nhạy cảnh báo)
Cách tính: (Số lần báo Nắng gắt ĐÚNG) chia cho (Tổng số lần thực tế có Nắng gắt).
Công thức văn bản: Recall = Số lần báo đúng / (Số lần đúng + Số lần bỏ sót).
Ý nghĩa: Tránh bỏ sót thiên tai. Đây là chỉ số quan trọng nhất cho sự an toàn. Nếu năm 2021 có 10 trận bão mà App chỉ báo được 2 trận, thì Recall chỉ đạt 20% (quá nguy hiểm).
######################################################################

KẾT LUẬN CUỐI CÙNG:
Một mô hình được coi là "Khả thi để đưa vào ứng dụng thực tế" khi và chỉ khi:
MAE nhiệt độ nhỏ hơn 2.0 độ C.
Recall cho các sự kiện thời tiết nguy hiểm (mưa lớn, nắng gắt) đạt trên 75%.
Sai số cực đại (Max Error) không vượt quá ngưỡng gây nguy hiểm cho người dùng trong các sự kiện thiên tai.
-----------------------------------------------------
- Biểu đồ linechart so sánh các mô hình với giá trị thực tế
- Thời tiết dự kiến của ngày của mô hình tốt nhất (cho chọn trong lịch) - thời tiết thực tế
- Ngưỡng (Nắng gắt / mưa lớn gây ngập/ gió mạnh gây bão) để cảnh báo người dùng


