# 🌿 Herb–Disease Association Prediction

## 📌 Giới thiệu
Dự án này dự đoán mối quan hệ giữa **thảo dược và bệnh** dựa trên các kỹ thuật **học sâu trên đồ thị**. Bài toán được mô hình hóa như một mạng lưới sinh học phức tạp, nơi thảo dược tác động đến bệnh thông qua nhiều hợp chất và protein.

## 🧠 Ý tưởng chính
Mô hình khai thác **hai nguồn thông tin**:
- Đồ thị tương tác (liên kết đã biết)
- Đồ thị tương đồng (quan hệ tiềm ẩn)

Kết hợp với:
- Attention (tập trung thông tin quan trọng)  
- Contrastive learning (cải thiện chất lượng biểu diễn)

## 🏗️ Phương pháp
- Sử dụng **Graph Attention Network (GAT)** để học biểu diễn
- Kiến trúc **đa góc nhìn (dual-view)**
- Kết hợp **contrastive loss** và **supervised loss**

## 📊 Dữ liệu (nằm trong folder data)
Dữ liệu từ TCM, gồm:
- Herb, Compound, Protein, Disease  

Tiền xử lý:
- Làm sạch, chuẩn hóa  
- Xây dựng đồ thị và tính similarity  

## ⚙️ Cài đặt
pip install -r requirements.txt

## ▶️ Chạy
- run_ablation.py (chạy thử nghiệm tất cả các cấu hình để chứng minh đóng góp của từng thành phần)
- run_tuning_proposed.py (chạy tuning tham số cho cấu hình đề xuất)
- proposed.py (chạy riêng cấu hình đề xuất)
- proposed_full_data_inference.py (train cấu hình đề xuất trên toàn bộ data để phục vụ nghiên cứu điển hình)
=> kết quả lưu trữ tại folder outputs

## 🧪 Thực nghiệm
- 5-fold cross-validation  
- Negative sampling  
- Optimizer: Adam  

## 📈 Đánh giá
- ROC-AUC  
- AUPR  
- F1-score  

## ⚠️ Lưu ý
- Nếu dùng CPU → tắt AMP  
- GPU sẽ giúp tăng tốc đáng kể  
