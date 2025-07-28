# app.py
import streamlit as st
from auth import login_page
from classroom import classroom_app

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    classroom_app()
else:
    login_page()


# auth.py
import streamlit as st

def login_page():
    st.title("\U0001F510 Đăng nhập vào phần mềm")

    if "account" not in st.session_state:
        st.session_state.account = {}

    tab1, tab2 = st.tabs(["Đăng nhập", "Đăng ký"])

    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Mật khẩu", type="password", key="login_pass")
        if st.button("Đăng nhập"):
            acc = st.session_state.account.get(email)
            if acc and acc["password"] == password:
                st.success("✅ Đăng nhập thành công!")
                st.session_state.logged_in = True
                st.session_state.current_user = email
                st.experimental_rerun()
            else:
                st.error("❌ Sai tài khoản hoặc mật khẩu.")

    with tab2:
        email = st.text_input("Email", key="register_email")
        password = st.text_input("Tạo mật khẩu", type="password", key="register_pass")
        name = st.text_input("Họ tên giáo viên")
        school = st.text_input("Trường")
        ward = st.text_input("Xã/Phường")

        if st.button("Đăng ký"):
            if email in st.session_state.account:
                st.warning("⚠️ Email đã được đăng ký.")
            else:
                st.session_state.account[email] = {
                    "password": password,
                    "name": name,
                    "school": school,
                    "ward": ward
                }
                st.success("🎉 Đăng ký thành công! Vui lòng đăng nhập.")


# classroom.py
import streamlit as st
import pandas as pd
import google.generativeai as genai

genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")

def classroom_app():
    st.title("📚 Quản lý lớp học bằng AI")

    user_email = st.session_state.current_user
    teacher = st.session_state.account[user_email]

    st.markdown(f"👩‍🏫 **{teacher['name']}** - *{teacher['school']}, {teacher['ward']}*")
    st.button("Đăng xuất", on_click=lambda: logout())

    st.header("1. Tạo lớp học")
    class_name = st.text_input("Tên lớp học")

    st.header("2. Nhập danh sách học sinh từ Excel")
    uploaded_file = st.file_uploader("Tải file Excel", type=["xlsx"])
    students_df = None
    if uploaded_file:
        students_df = pd.read_excel(uploaded_file)
        st.dataframe(students_df)

    st.header("3. Lập kế hoạch tuần bằng AI")
    week_topic = st.text_input("Chủ đề tuần")
    if st.button("Tạo kế hoạch"):
        if week_topic:
            response = model.generate_content(
                f"Lập kế hoạch dạy học tuần với chủ đề: {week_topic}. "
                "Trình bày theo bảng gồm: Thứ, Nội dung, Hoạt động, Ghi chú."
            )
            st.markdown(response.text)
        else:
            st.warning("Vui lòng nhập chủ đề.")

    st.header("4. Soạn thông báo gửi phụ huynh")
    message = st.text_area("Nội dung cần thông báo")
    if st.button("Soạn thông báo"):
        if message:
            response = model.generate_content(
                f"Tạo một thông báo ngắn gọn, lịch sự gửi phụ huynh: {message}"
            )
            st.info(response.text)
        else:
            st.warning("Vui lòng nhập nội dung.")

    st.header("5. Phân tích kết quả học tập")
    if students_df is not None and st.button("Phân tích kết quả"):
        result = model.generate_content(
            f"Phân tích kết quả học tập và hành vi của học sinh từ dữ liệu sau:\n\n{students_df.to_string(index=False)}"
        )
        st.markdown(result.text)
    elif students_df is None:
        st.info("Bạn cần tải file Excel trước.")

def logout():
    st.session_state.logged_in = False
    st.session_state.current_user = ""
    st.experimental_rerun()


# requirements.txt
streamlit>=1.34.0
pandas>=2.2.0
openpyxl>=3.1.2
google-generativeai>=0.5.0