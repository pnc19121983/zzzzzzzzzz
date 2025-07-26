import streamlit as st
import google.generativeai as genai

# Cấu hình Gemini
genai.configure(api_key=st.secrets["AIzaSyACFWxsjhnTruV05ap7-aSp_9DDQavGvHw"])
model = genai.GenerativeModel("gemini-1.5-flash")

# Danh sách độ khó
levels = ["Dễ", "Trung bình", "Khó"]

# Tạo câu hỏi từ AI
def generate_question(subject, grade, level):
    prompt = f"""
    Tạo 1 câu hỏi trắc nghiệm {level.lower()} cho học sinh lớp {grade}, môn {subject}.
    Bao gồm 4 lựa chọn (A, B, C, D) và đánh dấu đáp án đúng rõ ràng ở cuối.
    """
    response = model.generate_content(prompt)
    return response.text

# Kiểm tra đáp án đúng từ văn bản AI
def extract_answer(text):
    for line in text.split('\n'):
        if "Đáp án" in line:
            return line.strip().split(":")[-1].strip().upper()
    return None

# Streamlit UI
st.title("🎯 Bài kiểm tra thích ứng bằng Gemini AI")

# Bước 1: Nhập thông tin cá nhân
if "started" not in st.session_state:
    st.session_state.started = False
    st.session_state.current_q = 0
    st.session_state.score = 0
    st.session_state.difficulty = 0  # 0: Dễ, 1: TB, 2: Khó
    st.session_state.max_questions = 5
    st.session_state.quiz_log = []

if not st.session_state.started:
    name = st.text_input("👤 Họ và tên")
    school = st.text_input("🏫 Trường")
    grade = st.selectbox("🎓 Lớp", list(range(1, 13)))
    subject = st.selectbox("📘 Môn học", ["Toán", "Lý", "Hóa", "Tiếng Anh", "Sinh"])

    if st.button("🚀 Bắt đầu làm bài") and name and school:
        st.session_state.name = name
        st.session_state.school = school
        st.session_state.grade = grade
        st.session_state.subject = subject
        st.session_state.started = True
        st.experimental_rerun()

# Bước 2: Làm bài
else:
    st.markdown(f"👤 **Học sinh:** {st.session_state.name} | 🏫 **Trường:** {st.session_state.school}")
    st.markdown(f"📘 **Môn:** {st.session_state.subject} | 🎓 **Lớp:** {st.session_state.grade}")
    st.markdown(f"🔢 **Câu {st.session_state.current_q + 1} / {st.session_state.max_questions}**")

    if "current_question_text" not in st.session_state:
        q_text = generate_question(st.session_state.subject, st.session_state.grade, levels[st.session_state.difficulty])
        st.session_state.current_question_text = q_text
        st.session_state.correct_answer = extract_answer(q_text)

    question_lines = [line for line in st.session_state.current_question_text.split("\n") if line.strip() and "Đáp án" not in line]
    question_text = "\n".join(question_lines)
    st.markdown(question_text)

    answer = st.radio("Chọn đáp án của bạn:", ["A", "B", "C", "D"], key=st.session_state.current_q)

    if st.button("📨 Nộp câu trả lời"):
        correct = st.session_state.correct_answer
        user = answer.upper()

        # Ghi log
        st.session_state.quiz_log.append({
            "question": st.session_state.current_question_text,
            "your_answer": user,
            "correct_answer": correct
        })

        # Chấm điểm
        if user == correct:
            st.success("✅ Chính xác!")
            st.session_state.score += 1
            if st.session_state.difficulty < 2:
                st.session_state.difficulty += 1
        else:
            st.error(f"❌ Sai! Đáp án đúng là {correct}")
            if st.session_state.difficulty > 0:
                st.session_state.difficulty -= 1

        # Chuẩn bị câu tiếp
        st.session_state.current_q += 1
        del st.session_state.current_question_text  # xóa để tạo mới
        del st.session_state.correct_answer

        # Kết thúc hay chưa?
        if st.session_state.current_q >= st.session_state.max_questions:
            st.session_state.started = False
            st.success("🎉 Bạn đã hoàn thành bài kiểm tra!")
            st.markdown(f"**🔢 Số câu đúng: {st.session_state.score} / {st.session_state.max_questions}**")

            with st.expander("📋 Xem lại chi tiết câu hỏi"):
                for i, log in enumerate(st.session_state.quiz_log):
                    st.markdown(f"**Câu {i+1}:**")
                    st.markdown(log["question"])
                    st.markdown(f"🔹 Bạn chọn: `{log['your_answer']}` | ✅ Đáp án đúng: `{log['correct_answer']}`")
                    st.markdown("---")

            # Reset nút bắt đầu lại
            if st.button("🔁 Làm lại từ đầu"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.experimental_rerun()
        else:
            st.experimental_rerun()
