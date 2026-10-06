# Audit nguồn Kaggle

Nguồn: https://www.kaggle.com/code/chibuzorokocha/airline-sentiment-analysis-90-accuracy-using-rnn, Version 1, scriptVersionId 120745701. Code và output lịch sử đã được đọc; file nguồn lưu trong `sources/`, không thực thi LSTM trong project này.

- Mô hình nguồn: LSTM196, Embedding4000×128, SpatialDropout0.5; recurrent/input dropout0.3; head Dropout0.2 → Dense100/ReLU → Dropout0.4 → Dense2/softmax.
- Nguồn bỏ neutral và 5.000 negative đầu, còn 6.541 tweet; tokenizer fit toàn bộ subset trước split; train3662/val916/test1963; Adam (LR không tường minh), batch32,20epochs, không ES/checkpoint.
- Output nguồn: test accuracy 90.320940%, loss 0.652300; best val accuracy 92.58% ở epoch11. Đây không phải kết quả RNN nhóm.
- A1 sử dụng source cleaner/ordering/padding/preprocessing, nhưng RNN/tanh thay LSTM, seed42 và Adam0.001 tường minh là lựa chọn triển khai mới. Tokenizer tương đương Keras đã được kiểm tra trên toàn6.541tweet; maxlen tính lại=31. Không gọi A1 exact reproduction.
- Trong split A1 tạo lại, có 68 tweet IDs và 85 chuỗi clean text xuất hiện ở hơn một split. Đây là kiểm tra overlap trên file hiện có, không đo mức tăng metric do leakage.
- Phiên bản TensorFlow/Keras/CPU/RAM lịch sử không đủ thông tin. Environment hiện tại lưu riêng, không gán version mới cho output cũ.
