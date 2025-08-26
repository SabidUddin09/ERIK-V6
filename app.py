import streamlit as st
import requests
from bs4 import BeautifulSoup
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import fitz  # PyMuPDF
import docx
from googlesearch import search

# ------------------ ERIK v8 ------------------
st.set_page_config(page_title="ERIK v8 - AI Academic Assistant", layout="wide")
st.title("🧠 ERIK v8 - Exceptional Resources & Intelligence Kernel")

# ------------------ Sidebar ------------------
st.sidebar.header("Features")
mode = st.sidebar.radio("Choose a feature:", [
    "Ask Question",
    "Bangla Q&A",
    "Math Solver",
    "LaTeX Math Solver",
    "Scientific Calculator",
    "Quiz Generator",
    "PDF/Text Analyzer",
    "Graph Generator",
    "Google Scholar Search"
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

# ------------------ LaTeX Math Solver ------------------
elif mode == "LaTeX Math Solver":
    st.subheader("LaTeX Math Solver")
    latex_input = st.text_input("Enter equation (e.g., x^2 - 4 = 0):")
    if st.button("Solve in LaTeX"):
        try:
            eq = sp.sympify(latex_input.replace("^","**"))
            sol = sp.solve(eq, dict=True)
            if sol:
                for s in sol:
                    st.latex(f"x = {sp.latex(list(s.values())[0])}")
            else:
                st.warning("No solution found.")
        except Exception as e:
            st.error(f"Invalid equation: {e}")

# ------------------ Scientific Calculator ------------------
elif mode == "Scientific Calculator":
    st.subheader("Full Scientific Calculator")
    buttons = [
        ["7","8","9","/","sin","cos","tan"],
        ["4","5","6","*","asin","acos","atan"],
        ["1","2","3","-","log","ln","exp"],
        ["0",".","=","+","sqrt","^","!"],
        ["(",")","pi","e","C","Ans"]
    ]

    if 'calc_input' not in st.session_state:
        st.session_state.calc_input = ""
    if 'last_ans' not in st.session_state:
        st.session_state.last_ans = ""

    def press(btn):
        if btn == "=":
            try:
                st.session_state.last_ans = str(sp.sympify(st.session_state.calc_input).evalf())
                st.session_state.calc_input = st.session_state.last_ans
            except:
                st.session_state.calc_input = "Error"
        elif btn == "C":
            st.session_state.calc_input = ""
        elif btn == "Ans":
            st.session_state.calc_input += st.session_state.last_ans
        else:
            st.session_state.calc_input += btn

    st.text_area("Input", st.session_state.calc_input, height=50)
    for row in buttons:
        cols = st.columns(len(row))
        for i, b in enumerate(row):
            if cols[i].button(b):
                press(b)

# ------------------ Quiz Generator ------------------
elif mode == "Quiz Generator":
    st.subheader("Smart Quiz Generator")
    source_type = st.radio("Select source type:", ["Google Search", "PDF/Text Upload"])
    questions = []

    if source_type == "Google Search":
        topic = st.text_input("Enter topic for quiz generation:")
        num_q = st.number_input("Number of questions:", min_value=1, max_value=20, value=5)
        if st.button("Generate Quiz from Google"):
            st.info("Searching Google and generating questions...")
            try:
                content = ""
                for url in search(topic, num_results=5):
                    r = requests.get(url, timeout=3)
                    soup = BeautifulSoup(r.text, 'html.parser')
                    paragraphs = soup.find_all('p')
                    for p in paragraphs[:5]:
                        content += p.get_text() + " "
                sentences = content.split('. ')
                for i in range(min(num_q, len(sentences))):
                    q_text = sentences[i].strip()[:100] + "..."
                    st.write(f"Q{i+1}: {q_text}")
                    st.write("a) Option A  b) Option B  c) Option C  d) Option D")
            except Exception as e:
                st.error(f"Error generating quiz: {e}")

    elif source_type == "PDF/Text Upload":
        uploaded_file = st.file_uploader("Upload PDF / DOCX / TXT file", type=['pdf','txt','docx'])
        num_q = st.number_input("Number of questions:", min_value=1, max_value=20, value=5, key="pdfquiz")
        if uploaded_file and st.button("Generate Quiz from PDF/Text"):
            text = ""
            try:
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
                sentences = text.split('. ')
                for i in range(min(num_q, len(sentences))):
                    q_text = sentences[i].strip()[:100] + "..."
                    st.write(f"Q{i+1}: {q_text}")
                    st.write("a) Option A  b) Option B  c) Option C  d) Option D")
            except Exception as e:
                st.error(f"Error reading file or generating quiz: {e}")

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

# ------------------ Graph Generator ------------------
elif mode == "Graph Generator":
    st.subheader("Generate Graphs")
    raw_input = st.text_input(
        "Enter function (2D: x only, 3D: use x and y, e.g., sin(x)*cos(y)):")
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
            else:
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

# ------------------ Google Scholar Search ------------------
elif mode == "Google Scholar Search":
    st.subheader("Search Google Scholar")
    keyword = st.text_input("Enter research topic:")
    if st.button("Search Scholar"):
        st.info("Fetching top research papers...")
        query = f"{keyword} site:scholar.google.com"
        results = []
        try:
            for url in search(query, num_results=5):
                results.append(url)
        except:
            st.error("Error searching Scholar.")

        if results:
            for r in results:
                try:
                    r_page = requests.get(r, timeout=3)
                    soup = BeautifulSoup(r_page.text, "html.parser")
                    title = soup.title.string if soup.title else r
                    st.markdown(f"[{title}]({r})")
                except:
                    st.markdown(f"[{r}]({r})")
        else:
            st.warning("No papers found. Try a different keyword.")
