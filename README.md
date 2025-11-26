**Bộ dữ liệu**

*   Chi tiết: [Google Drive](https://drive.google.com/drive/folders/1ghoi12lbgHkcK32BCAHcbnlZvrlaZK5Y?usp=sharing)
*   Chú thích:
    *   `train.csv`: Dữ liệu huấn luyện.
    *   `test.csv`: Dữ liệu kiểm tra.
    *   `validation.csv`: Dữ liệu xác thực.
*   Hình ảnh: Huấn luyện, Kiểm tra, Xác thực (mỗi thư mục chứa các thư mục con cho nhãn)

**Mã nguồn**

*   `models`:
    *   MobileNet\_v2 + LSTM (có Attention)
    *   MobileNet\_v2 + LSTM (không có Attention)
    *   CNN + LSTM (có Attention)
    *   CNN + LSTM (không có Attention)
*   `data_generation`:
    *   `question_generate_v2.py`: Tạo câu hỏi từ hình ảnh bằng Gemini API.
    *   `preprocess.ipynb`: Tiền xử lý dữ liệu thô.

**Tài liệu**

*   `slide.pdf`
*   `report.pdf`
