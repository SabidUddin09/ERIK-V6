import streamlit as st
import requests
from bs4 import BeautifulSoup
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import fitz  # PyMuPDF
import docx
from googlesearch import search

# ------------------ ERIK v6 ------------------
st.set_page_config(page_title="ERIK v6 - AI Academic Assistant", layout="wide")
st.title("🧠 ERIK v6 - Exceptional Resources & Intelligence Kernel")

# ------------------ Sidebar ------------------
st.sidebar.header("Features")
mode = st.sidebar.radio("Choose a feature:", [
    "Ask Question",
    "Bangla Q&A",
    "Math Solver",
    "Scientific Calculator",
    "Quiz Generator",
    "PDF/Text Analyzer",
    "Graph Generator"
])

# ------------------ Ask Question ------------------
if mode == "Ask Question":
    query = st.text_input("Ask anything (English):")
    if st.button("Search & Answer"):
        st.info("Searching Google...")
        results = []
        try:
            for url in search(query, num_results=5):
                results.append(url)
        except:
            st.error("Error searching Google.")

        answer = ""
        for link in results:
            try:
                r = requests.get(link, timeout=3)
                soup = BeautifulSoup(r.text, 'html.parser')
                paragraphs = soup.find_all('p')
                for p in paragraphs[:2]:
                    answer += p.get_text() + "\n"
            except:
                continue

        if answer:
            st.markdown("**Answer from web sources:**")
            st.write(answer)
            st.markdown("**Top sources:**")
            for r in results:
                st.write(f"- {r}")
        else:
            st.warning("No answer found. Try rephrasing the question.")

# ------------------ Bangla Q&A ------------------
elif mode == "Bangla Q&A":
    query = st.text_input("যেকোনো প্রশ্ন করুন (বাংলায়):")
    if st.button("Search in Bangla"):
        st.info("গুগলে সার্চ করা হচ্ছে...")
        results = []
        try:
            for url in search(query, num_results=5, lang="bn"):
                results.append(url)
        except:
            st.error("Error searching Google.")

        answer = ""
        for link in results:
            try:
                r = requests.get(link, timeout=3)
                soup = BeautifulSoup(r.text, 'html.parser')
                paragraphs = soup.find_all('p')
                for p in paragraphs[:2]:
                    answer += p.get_text() + "\n"
            except:
                continue

        if answer:
            st.markdown("**ওয়েব থেকে উত্তর:**")
            st.write(answer)
            st.markdown("**উৎস লিঙ্কসমূহ:**")
            for r in results:
                st.write(f"- {r}")
        else:
            st.warning("কোনো উত্তর পাওয়া যায়নি। প্রশ্নটি অন্যভাবে লিখে চেষ্টা করুন।")

# ------------------ Math Solver ------------------
elif mode == "Math Solver":
    st.subheader("Math Problem Solver")
    problem = st.text_area("Enter a math problem (e.g., x**2 - 4):")
    if st.button("Solve"):
        try:
            x = sp.symbols('x')
            solution = sp.solve(problem, x)
            st.success(f"Solution: {solution}")
        except Exception as e:
            st.error(f"Error: {e}")

# ------------------ Scientific Calculator with Buttons ------------------
elif mode == "Scientific Calculator":
    st.subheader("Scientific Calculator")
    
    # Define calculator buttons
    buttons = [
        ["7","8","9","/","sin"],
        ["4","5","6","*","cos"],
        ["1","2","3","-","tan"],
        ["0",".","=","+","log"],
        ["(",")","pi","**","sqrt"]
    ]
    
    if 'calc_input' not in st.session_state:
        st.session_state.calc_input = ""

    def press(btn):
        if btn == "=":
            try:
                st.session_state.calc_input = str(sp.sympify(st.session_state.calc_input).evalf())
            except:
                st.session_state.calc_input = "Error"
        else:
            st.session_state.calc_input += btn

    # Display current input
    st.text_area("Input", st.session_state.calc_input, height=50)

    # Display buttons
    for row in buttons:
        cols = st.columns(len(row))
        for i, button in enumerate(row):
            if cols[i].button(button):
                press(button)
    
    if st.button("Clear"):
        st.session_state.calc_input = ""

# ------------------ Quiz Generator ------------------
elif mode == "Quiz Generator":
    st.subheader("Generate Multiple Choice Questions")
    topic = st.text_input("Enter topic:")
    num_q = st.number_input("Number of questions:", min_value=1, max_value=20, value=5)
    if st.button("Generate Quiz"):
        for i in range(num_q):
            st.write(f"Q{i+1}: This is a placeholder question about {topic}?")
            st.write("a) Option A  b) Option B  c) Option C  d) Option D")

# ------------------ PDF/Text Analyzer ------------------
elif mode == "PDF/Text Analyzer":
    st.subheader("Upload PDF or TXT/DOCX")
    uploaded_file = st.file_uploader("Choose a file", type=['pdf','txt','docx'])
    if uploaded_file:
        text = ""
        if uploaded_file.type == "application/pdf":
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in doc:
                text += page.get_text()
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            text = str(uploaded_file.read(), "utf-8")
        st.text_area("Extracted Text", text, height=300)

# ------------------ Graph Generator with Smart Equation Writer ------------------
elif mode == "Graph Generator":
    st.subheader("Generate Graphs")
    raw_input = st.text_input(
        "Enter function (2D: x only, 3D: use x and y, e.g., sin(x)*cos(y)):")

    # Auto-format smart equation
    func_input = raw_input.replace("^","**").replace("X","x").replace("Y","y")
    
    graph_type = st.radio("Graph Type:", ["2D", "3D"])
    if st.button("Plot Graph"):
        try:
            if graph_type == "2D":
                x = sp.symbols('x')
                func = sp.sympify(func_input)
                x_vals = np.linspace(-10, 10, 400)
                y_vals = [func.subs(x, val) for val in x_vals]
                plt.plot(x_vals, y_vals)
                plt.xlabel("x")
                plt.ylabel("y")
                plt.title(f"Graph of {func_input}")
                st.pyplot(plt)
            else:  # 3D
                x, y = sp.symbols('x y')
                func = sp.sympify(func_input)
                X = np.linspace(-5, 5, 100)
                Y = np.linspace(-5, 5, 100)
                X, Y = np.meshgrid(X, Y)
                Z_func = sp.lambdify((x, y), func, "numpy")
                Z = Z_func(X, Y)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(X, Y, Z, cmap='viridis')
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                plt.title(f"3D Graph of {func_input}")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Invalid equation: {e}")
