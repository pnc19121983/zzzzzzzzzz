import streamlit as st
import google.generativeai as genai
import json
import pandas as pd
import time
import os

# Cấu hình Gemini
try:
    genai.configure(api_key=st.secrets["AIzaSyACFWxsjhnTruV05ap7-aSp_9DDQavGvHw"])
    model = genai.GenerativeModel("gemini-1.5-flash")
except KeyError:
    st.error("❌ Không tìm thấy API Key. Vui lòng cấu hình trong Streamlit Secrets.")
    st.stop()

# Danh sách độ khó
LEVELS = ["Dễ", "Trung bình", "Khó", "Rất khó"]

# Tạo câu hỏi từ Gemini
def generate_question(subject, grade, level, retries=3):
    prompt = f"""
    Tạo 1 câu hỏi trắc nghiệm {level.lower()} cho học sinh lớp {grade}, môn {subject}, theo chương trình giáo dục phổ thông 2018 của Việt Nam.
    Trả về định dạng JSON:
    {{
      "question": "Nội dung câu hỏi",
      "options": {{
        "A": "Lựa chọn A",
        "B": "Lựa chọn B",
        "C": "Lựa chọn C",
        "D": "Lựa chọn D"
      }},
      "answer": "A"
    }}
    Câu hỏi rõ ràng, đúng chính tả, phù hợp với chương trình học.
    """
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
                continue
            st.error(f"❌ Lỗi khi gọi Gemini sau {retries} lần thử: {e}")
            return {
                "question": "Không thể tạo câu hỏi. Vui lòng thử lại.",
                "options": {"A": "Lựa chọn A", "B": "Lựa chọn B", "C": "Lựa chọn C", "D": "Lựa chọn D"},
                "answer": "A"
            }

# Lưu kết quả vào CSV
def save_results(name, school, subject, grade, quiz_log, score, max_questions):
    data = [{
        "Câu": i + 1,
        "Câu hỏi": log["question"],
        "Lựa chọn": "; ".join([f"{k}: {v}" for k, v in log["options"].items()]),
        "Đáp án của bạn": log["your_answer"],
        "Đáp án đúng": log["correct_answer"],
        "Kết quả": "Đúng" if log["your_answer"] == log["correct_answer"] else "Sai"
    } for i, log in enumerate(quiz_log)]
    df = pd.DataFrame(data)
    filename = f"results_{name}_{subject}_grade{grade}_{int(time.time())}.csv"
    df.to_csv(filename, index=False, encoding="utf-8")
    return filename

# UI chính
st.title("🎯 KIỂM TRA NĂNG LỰC")

# Trạng thái ban đầu
if "started" not in st.session_state:
    st.session_state.started = False
    st.session_state.current_q = 0
    st.session_state.score = 0
    st.session_state.difficulty = 0
    st.session_state.quiz_log = []
    st.session_state.timer_start = None

# Bước 1: Nhập thông tin
if not st.session_state.started:
    st.subheader("📋 Nhập thông tin")
    name = st.text_input("👤 Họ và tên", key="name")
    school = st.text_input("🏫 Trường", key="school")
    grade = st.selectbox("🎓 Lớp", list(range(1, 13)), key="grade")
    subject = st.selectbox("📘 Môn học", ["Toán", "Tiếng Việt", "Tiếng Anh", "Tin học", "Vật lí", "Hóa học", "Sinh học", "Lịch sử", "Địa lí"], key="subject")
    max_questions = st.number_input("🔢 Số câu hỏi", min_value=5, max_value=50, value=15, key="max_questions")

    if st.button("🚀 Bắt đầu làm bài"):
        if name and school:
            st.session_state.name = name
            st.session_state.school = school
            st.session_state.grade = grade
            st.session_state.subject = subject
            st.session_state.max_questions = max_questions
            st.session_state.started = True
            st.session_state.timer_start = time.time()
            st.rerun()
        else:
            st.warning("⚠️ Vui lòng nhập đầy đủ họ tên và trường.")

# Bước 2: Làm bài
else:
    st.markdown(f"👤 **Học sinh:** {st.session_state.name} | 🏫 **Trường:** {st.session_state.school}")
    st.markdown(f"📘 **Môn:** {st.session_state.subject} | 🎓 **Lớp:** {st.session_state.grade}")
    st.markdown(f"🔢 **Câu {st.session_state.current_q + 1} / {st.session_state.max_questions}**")
    st.progress(st.session_state.current_q / st.session_state.max_questions)

    # Hiển thị thời gian
    if st.session_state.timer_start:
        elapsed_time = time.time() - st.session_state.timer_start
        st.markdown(f"⏱ **Thời gian làm bài:** {int(elapsed_time // 60)} phút {int(elapsed_time % 60)} giây")

    # Tạo câu hỏi
    if "current_question" not in st.session_state:
        st.session_state.current_question = generate_question(
            st.session_state.subject,
            st.session_state.grade,
            LEVELS[st.session_state.difficulty]
        )

    question_data = st.session_state.current_question
    st.markdown(f"#### ❓ {question_data['question']}")
    answer = st.radio(
        "🔘 Chọn đáp án của bạn:",
        question_data["options"].keys(),
        format_func=lambda x: f"{x}. {question_data['options'][x]}",
        key=f"q_{st.session_state.current_q}"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📨 Nộp câu trả lời"):
            correct = question_data["answer"].upper()
            user = answer.upper()

            st.session_state.quiz_log.append({
                "question": question_data["question"],
                "options": question_data["options"],
                "your_answer": user,
                "correct_answer": correct
            })

            if user == correct:
                st.success("✅ Chính xác!")
                st.session_state.score += 1
                if st.session_state.difficulty < len(LEVELS) - 1:
                    st.session_state.difficulty += 1
            else:
                st.error(f"❌ Sai! Đáp án đúng là {correct}. {question_data['options'][correct]}")
                if st.session_state.difficulty > 0:
                    st.session_state.difficulty -= 1

            st.session_state.current_q += 1
            del st.session_state.current_question

            if st.session_state.current_q >= st.session_state.max_questions:
                st.session_state.started = False
                filename = save_results(
                    st.session_state.name,
                    st.session_state.school,
                    st.session_state.subject,
                    st.session_state.grade,
                    st.session_state.quiz_log,
                    st.session_state.score,
                    st.session_state.max_questions
                )
                st.success(f"🎉 Bạn đã hoàn thành bài kiểm tra! Kết quả đã được lưu vào `{filename}`.")
                st.markdown(f"**✅ Số câu đúng: {st.session_state.score} / {st.session_state.max_questions}**")
                st.markdown(f"**📊 Tỷ lệ đúng: {st.session_state.score / st.session_state.max_questions * 100:.2f}%**")

                with st.expander("📋 Xem lại chi tiết câu hỏi"):
                    for i, log in enumerate(st.session_state.quiz_log):
                        st.markdown(f"**Câu {i+1}:** {log['question']}")
                        for opt, text in log["options"].items():
                            st.markdown(f"- {opt}. {text}")
                        st.markdown(f"🔹 Bạn chọn: `{log['your_answer']}` | ✅ Đáp án đúng: `{log['correct_answer']}`")
                        st.markdown("---")

                if st.button("🔁 Làm lại từ đầu"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
            else:
                st.rerun()

    with col2:
        if st.button("⬅️ Quay lại chỉnh sửa thông tin"):
            for key in ["name", "school", "grade", "subject", "max_questions"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.started = False
            st.rerun()