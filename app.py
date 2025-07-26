import streamlit as st
import google.generativeai as genai
import re

# ✅ Cấu hình Gemini – dán trực tiếp API Key
genai.configure(api_key="AIzaSyACFWxsjhnTruV05ap7-aSp_9DDQavGvHw")  # <-- Thay YOUR_API_KEY_HERE bằng key của bạn
model = genai.GenerativeModel("gemini-1.5-flash")

# ✅ Danh sách độ khó
levels = ["Dễ", "Trung bình", "Khó", "Rất khó"]

# ✅ Tạo câu hỏi từ Gemini
def generate_question(subject, grade, level):
    prompt = f"""
    Tạo 1 câu hỏi trắc nghiệm {level.lower()} cho học sinh lớp {grade}, môn {subject}.
    Bao gồm 4 lựa chọn (A, B, C, D) và đánh dấu đáp án đúng rõ ràng ở cuối dưới dạng "Đáp án: A". Các câu hỏi rõ ràng, chữ viết đúng chính tả, chuẩn mực, đúng theo chương trình giáo dục phổ thông 2018 của Việt Nam
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"❌ Lỗi khi gọi Gemini: {e}")
        return "Không thể tạo câu hỏi. Vui lòng thử lại sau."

# ✅ Trích xuất đáp án đúng từ nội dung
def extract_answer(text):
    for line in text.split('\n'):
        if "Đáp án" in line:
            return line.strip().split(":")[-1].strip().upper()
    return "A"

# ✅ Phân tích câu hỏi & lựa chọn
def parse_question(text):
    lines = text.strip().split("\n")
    lines = [l for l in lines if l.strip() and "Đáp án" not in l]
    joined = " ".join(lines)
    
    pattern = r"^(.*?)(?:\s*)A[.\):\-]?\s*(.*?)\s*B[.\):\-]?\s*(.*?)\s*C[.\):\-]?\s*(.*?)\s*D[.\):\-]?\s*(.*)"
    match = re.match(pattern, joined, re.IGNORECASE)

    if match:
        question = match.group(1).strip()
        options = {
            "A": match.group(2).strip(),
            "B": match.group(3).strip(),
            "C": match.group(4).strip(),
            "D": match.group(5).strip(),
        }
        return question, options
    else:
        return joined, {
            "A": "Lựa chọn A",
            "B": "Lựa chọn B",
            "C": "Lựa chọn C",
            "D": "Lựa chọn D"
        }

# ✅ UI chính
st.title("🎯 KIỂM TRA NĂNG LỰC")

# ✅ Trạng thái ban đầu
if "started" not in st.session_state:
    st.session_state.started = False
    st.session_state.current_q = 0
    st.session_state.score = 0
    st.session_state.difficulty = 0
    st.session_state.max_questions = 15
    st.session_state.quiz_log = []

# ✅ Bước 1: Nhập thông tin
if not st.session_state.started:
    name = st.text_input("👤 Họ và tên")
    school = st.text_input("🏫 Trường")
    grade = st.selectbox("🎓 Lớp", list(range(1, 13)))
    subject = st.selectbox("📘 Môn học", ["Toán", "Tiếng Việt", "Tiếng Anh", "Tin học","Vật lí", "Hóa học", "Sinh học", "Lịch sử", "Địa lí"])

    if st.button("🚀 Bắt đầu làm bài") and name and school:
        st.session_state.name = name
        st.session_state.school = school
        st.session_state.grade = grade
        st.session_state.subject = subject
        st.session_state.started = True
        st.rerun()

# ✅ Bước 2: Làm bài
else:
    st.markdown(f"👤 **Học sinh:** {st.session_state.name} | 🏫 **Trường:** {st.session_state.school}")
    st.markdown(f"📘 **Môn:** {st.session_state.subject} | 🎓 **Lớp:** {st.session_state.grade}")
    st.markdown(f"🔢 **Câu {st.session_state.current_q + 1} / {st.session_state.max_questions}**")

    # Tạo câu hỏi nếu chưa có
    if "current_question_text" not in st.session_state:
        q_text = generate_question(
            st.session_state.subject,
            st.session_state.grade,
            levels[st.session_state.difficulty]
        )
        st.session_state.current_question_text = q_text
        st.session_state.correct_answer = extract_answer(q_text)

    q_text = st.session_state.current_question_text
    question_text, options = parse_question(q_text)

    st.markdown(f"#### ❓ {question_text}")
    answer = st.radio(
        "🔘 Chọn đáp án của bạn:",
        options.keys(),
        format_func=lambda x: f"{x}. {options[x]}",
        key=st.session_state.current_q
    )

    # Nộp câu trả lời
    if st.button("📨 Nộp câu trả lời"):
        correct = st.session_state.correct_answer
        user = answer.upper()

        st.session_state.quiz_log.append({
            "question": q_text,
            "your_answer": user,
            "correct_answer": correct
        })

        if user == correct:
            st.success("✅ Chính xác!")
            st.session_state.score += 1
            if st.session_state.difficulty < 2:
                st.session_state.difficulty += 1
        else:
            st.error(f"❌ Sai! Đáp án đúng là {correct}")
            if st.session_state.difficulty > 0:
                st.session_state.difficulty -= 1

        st.session_state.current_q += 1
        del st.session_state.current_question_text
        del st.session_state.correct_answer

        if st.session_state.current_q >= st.session_state.max_questions:
            st.session_state.started = False
            st.success("🎉 Bạn đã hoàn thành bài kiểm tra!")
            st.markdown(f"**✅ Số câu đúng: {st.session_state.score} / {st.session_state.max_questions}**")

            with st.expander("📋 Xem lại chi tiết câu hỏi"):
                for i, log in enumerate(st.session_state.quiz_log):
                    st.markdown(f"**Câu {i+1}:**")
                    st.markdown(log["question"])
                    st.markdown(f"🔹 Bạn chọn: `{log['your_answer']}` | ✅ Đáp án đúng: `{log['correct_answer']}`")
                    st.markdown("---")

            if st.button("🔁 Làm lại từ đầu"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        else:
            st.rerun()
