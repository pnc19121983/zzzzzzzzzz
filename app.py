import streamlit as st
import pandas as pd
import google.generativeai as genai

# Cấu hình Gemini 1.5 Flash
genai.configure(api_key="AIzaSyBwr6QBpHRYSTMsMTm3KwLnM4GO2TotuP4")
model = genai.GenerativeModel("gemini-1.5-flash")

# Tiêu đề
st.title("📚 Quản lí lớp học bằng AI")

# 1. Đăng ký giáo viên
st.header("1. Đăng ký thông tin giáo viên")
with st.form("register_form"):
    name = st.text_input("Họ và tên giáo viên")
    school = st.text_input("Trường")
    ward = st.text_input("Xã / Phường")
    submitted = st.form_submit_button("Lưu thông tin")
    if submitted:
        st.success(f"Giáo viên {name} - {school}, {ward} đã đăng ký thành công.")

# 2. Tạo lớp học
st.header("2. Tạo lớp học")
class_name = st.text_input("Tên lớp học")

# 3. Nhập danh sách học sinh từ Excel
st.header("3. Nhập danh sách lớp từ Excel")
uploaded_file = st.file_uploader("Tải lên file Excel", type=["xlsx"])
students_df = None
if uploaded_file:
    students_df = pd.read_excel(uploaded_file)
    st.dataframe(students_df)

# 4. Xây dựng kế hoạch tuần bằng AI
st.header("4. Lập kế hoạch tuần bằng AI")
week_topic = st.text_input("Chủ đề / nội dung tuần")
if st.button("Tạo kế hoạch"):
    if week_topic:
        response = model.generate_content(
            f"Tôi là giáo viên. Hãy giúp tôi lập kế hoạch dạy học trong tuần về chủ đề: {week_topic}. "
            "Trình bày dạng bảng: Thứ, Nội dung, Hoạt động, Ghi chú."
        )
        st.markdown(response.text)
    else:
        st.warning("Hãy nhập chủ đề tuần.")

# 5. Tạo thông báo gửi phụ huynh
st.header("5. Tạo thông báo gửi phụ huynh")
message_content = st.text_area("Nội dung chính cần thông báo (ví dụ: nghỉ học, kiểm tra...)")
if st.button("Soạn thông báo"):
    if message_content:
        response = model.generate_content(
            f"Tạo một thông báo lịch sự, ngắn gọn gửi cho phụ huynh học sinh với nội dung: {message_content}"
        )
        st.info(response.text)
    else:
        st.warning("Nhập nội dung cần thông báo.")

# 6. Phân tích kết quả học tập
st.header("6. Phân tích kết quả học tập & định hướng")
if students_df is not None:
    if st.button("Phân tích kết quả học tập"):
        result = model.generate_content(
            f"Dựa trên dữ liệu sau đây, hãy phân tích tình hình học tập và rèn luyện, "
            f"và đưa ra định hướng cá nhân hoặc chung cho học sinh:\n\n{students_df.to_string(index=False)}"
        )
        st.markdown(result.text)
else:
    st.info("Vui lòng tải file Excel để phân tích.")

