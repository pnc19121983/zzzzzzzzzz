import streamlit as st
import google.generativeai as genai

# Cấu hình Gemini – dán trực tiếp API Key
genai.configure(api_key="AIzaSyACFWxsjhnTruV05ap7-aSp_9DDQavGvHw")
model = genai.GenerativeModel("gemini-1.5-flash")

# Danh sách độ khó
levels = ["Dễ", "Trung bình", "Khó"]

# Tạo câu hỏi từ AI
def generate_question(subject, grade, level):
    prompt = f"""
    Tạo 1 câu hỏi trắc nghiệm {level.lower()} cho học sinh lớp {grade}, môn {subject}.
    Bao gồm 4 lựa chọn (A, B, C, D) và đánh dấu đáp án đúng rõ ràng ở cuối dưới dạng "Đáp án: A".
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"❌ Lỗi khi tạo câu hỏi từ Gemini: {e}")
        return "Không thể tạo câu hỏi. Vui lòng thử lại sau."

# Tách đáp án đúng từ văn bản
def extract_answer(text):
    for line in text.split('\n'):
        if "Đáp án" in line:
            return line.strip().split(":")[-1].strip().upper()
    return "A"  # fallback nếu không tìm thấy đáp án

# Giao diện Streamlit
st.title("🎯 Bài kiểm tra thích ứng bằng Gemini AI")

# Khởi tạo trạng thái
if "started" not in st.session_state:
    st.session_state.started = False
    st.session_state.current_q = 0
    st.session_state.score = 0
    st.session_state.difficulty = 0
    st.session_state.max_questions = 5
    st.session_state.quiz_log = []

# Bước 1: Nhập thông tin cá nhân
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
        st.rerun()

# Bước 2: Làm bài
else:
    st.markdown(f"👤 **Học sinh:** {st.session_state.name} | 🏫 **Trường:** {st.session_state.school}")
    st.markdown(f"📘 **Môn:** {st.session_state.subject} | 🎓 **Lớp:** {st.session_state.grade}")
    st.markdown(f"🔢 **Câu {st.session_state.current_q + 1} / {st.session_state.max_questions}**")

    if "current_question_text" not in st.session_state:
        q_text = generate_question(
            st.session_state.subject,
            st.session_state.grade,
            levels[st.session_state.difficulty]
        )
        st.session_state.current_question_text = q_text
        st.session_state.correct_answer = extract_answer(q_text)

    # Hiển thị câu hỏi (ẩn dòng đáp án)
    question_lines = [
        line for line in st.session_state.current_question_text.split("\n")
        if line.strip() and "Đáp án" not in line
    ]
    question_text = "\n".join(question_lines)
    st.markdown(question_text)

    # Hiển thị debug nếu cần
    # st.expander("🔍 Xem nội dung AI trả về").markdown(st.session_state.current_question_text)

    answer = st.radio("Chọn đáp án của bạn:", ["A", "B", "C", "D"], key=st.session_state.current_q)

    if st.button("📨 Nộp câu trả lời"):
        correct = st.session_state.correct_answer
        user = answer.upper()

        st.session_state.quiz_log.append({
            "question": st.session_state.current_question_text,
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

            if st.button("🔁 Làm lại từ đầu"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        else:
            st.rerun()
