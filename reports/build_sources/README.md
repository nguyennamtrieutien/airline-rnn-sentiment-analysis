# Nguồn tạo báo cáo và slide

Các file này phục vụ biên tập/tạo lại tài liệu; không thuộc dependency cần để huấn luyện hay inference. Toàn bộ kết quả ML có thể chạy lại bằng pipeline ở README gốc. Có thể sửa trực tiếp DOCX/PPTX trong Word/PowerPoint.

## Word

`build_report.py` đọc CSV/JSON/hình đã có, template `sources/report_template.docx` và `page-map.json`. Cần Python với `python-docx`, `lxml`, `Pillow`. Template là bản sao nguyên vẹn file mẫu được người cung cấp xác nhận cho việc dùng thông tin bìa; không sửa file gốc ở thư mục Mẫu tham khảo.

Từ project root:

```bash
python reports/build_sources/build_report.py
```

Lệnh tạo lại `reports/final/BaoCao_Airline_RNN.docx` và hai JSON nguồn nội dung/tài liệu tham khảo. Bản tạo này có bảng/công thức native, mục lục liên kết nội bộ và số trang lấy từ page map đã kiểm tra trong lần bàn giao. Dùng bản sao nếu đã biên tập thủ công vì script tạo lại toàn bộ nội dung từ nguồn.

Sau khi thay nội dung ảnh hưởng số trang, render lại bằng Word hoặc trình render DOCX, cập nhật `page-map.json` theo số trang Arabic của main section rồi tạo lại. Sáu trang đầu thuộc front matter; mục lục dùng trang của nội dung chính, không dùng physical page. Mục lục hiện là bản đã điền số trang có bookmark hyperlinks, không phải TOC field tự tạo lại. Word có thể phân trang khác tùy font/printer; kiểm tra trang trước khi in/nộp.

## PowerPoint

`build_slides.mjs` dùng `@oai/artifact-tool` và helper của Presentations skill trong Codex. Đây là dependency authoring riêng của runtime, không có trong `requirements.txt` ML. Thiết lập các biến `RUNTIME_NODE_MODULES`, `PRESENTATIONS_SKILL_PATH`, `REPORT_PYTHON` theo môi trường Codex hiện tại; module resolver Node phải truy cập được `@oai/artifact-tool`. `AIRLINE_PROJECT_ROOT` tùy chọn, mặc định tìm project theo vị trí script.

```bash
node reports/build_sources/build_slides.mjs
```

Mặc định xuất `ThuyetTrinh_Airline_RNN_generated.pptx`, giữ bản bàn giao để đối chiếu. `DECK_NAME` tùy chọn đặt tên khác. Dùng tên mới cho mỗi lần finalize để không trùng validation receipt. Bảng và biểu đồ là thành phần native; workbook nhúng là snapshot của số liệu đã đọc, không có liên kết cập nhật trực tiếp từ CSV. Chart snapshot làm tròn sáu chữ số thập phân, CSV vẫn giữ số gốc. Speaker notes và các nguồn theo slide lưu trong `presentation_content.json`.

Preview/receipt trung gian nằm trong `.build/`, bị loại khỏi ZIP. Khi không có runtime Codex, biên tập PPTX trực tiếp bằng PowerPoint; không cần công cụ tạo để dùng tài liệu đã bàn giao. Render/import và kiểm tra package không thay cho thử trình chiếu trên Microsoft PowerPoint thực tế.

Ảnh demo toàn trang ở `F23_demo.png`; `scripts/prepare_demo_figure.py` tạo biến thể hai phần `F23_demo_report.png` dùng trong Word/slide từ ảnh chụp thực đã kiểm tra. Hai ảnh cùng một lượt inference; không thay đổi xác suất.
