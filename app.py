import streamlit as st
import requests
from bs4 import BeautifulSoup
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import fitz  # PyMuPDF
import docx
from googlesearch import search

# ------------------ ERIK v6 Full ------------------
st.set_page_config(page_title="ERIK v6 - AI Academic Assistant", layout="wide")

# ------------------ Introduction ------------------
st.title("🧠 ERIK v6 - Exceptional Resources & Intelligence Kernel")
st.markdown("""
**Welcome to ERIK v6!**  
ERIK is an AI-powered academic assistant designed to help students and researchers with **math, graphing, research, and quizzes**.  
✨ **Created and developed by Sabid Uddin Nahian**.  
Use the sidebar to explore the features! 🚀
""")

# ------------------ Sidebar ------------------
st.sidebar.header("Features")
mode = st.sidebar.radio("Choose a feature:", [
    "Ask Question",
    "Quiz Generator",
    "PDF/Text Analyzer",
    "LaTeX Math Solver",
    "Scientific Calculator (Desmos-like)",
    "Google Scholar Search",
    "2D & 3D Graph Generator"
])

# ------------------ Ask Question (Multilingual) ------------------
if mode == "Ask Question":
    st.subheader("🌐 Ask Question (English, Bangla, German & Mandarin)")
    language = st.selectbox("Select language:", ["English", "Bangla", "German", "Mandarin"])
    query = st.text_input("Enter your question:")
    if st.button("Search & Answer 🔍"):
        st.info("Searching the web... ⏳")
        lang_code = {"English":"en", "Bangla":"bn", "German":"de", "Mandarin":"zh-CN"}[language]
        results = []
        try:
            for url in search(query, num_results=5, lang=lang_code):
                results.append(url)
        except:
            st.error("❌ Error searching the web.")
        
        answer = ""
        for link in results:
            try:
                r = requests.get(link, timeout=3)
                soup = BeautifulSoup(r.text, 'html.parser')
                paragraphs = soup.find_all('p')
                for p in paragraphs[:2]:
                    answer += p.get_text() + " "
            except:
                continue
        
        if answer:
            st.markdown("💡 **Here’s what I found for you:**")
            st.write(f"🤖 {answer.strip()}\n\n✨ Hope this helps! Keep exploring! 🚀")
            st.markdown("🔗 **Top sources:**")
            for r in results:
                st.write(f"- {r}")
        else:
            st.warning("⚠️ Hmm… I couldn't find an answer. Try asking differently!")

# ------------------ Quiz Generator ------------------
elif mode == "Quiz Generator":
    st.subheader("📝 Smart Quiz Generator")
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
                st.error(f"❌ Error generating quiz: {e}")

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
                st.error(f"❌ Error reading file or generating quiz: {e}")

# ------------------ PDF/Text Analyzer ------------------
elif mode == "PDF/Text Analyzer":
    st.subheader("📄 PDF/Text Analyzer")
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

# ------------------ LaTeX Math Solver ------------------
elif mode == "LaTeX Math Solver":
    st.subheader("✍️ LaTeX Math Solver")
    latex_input = st.text_input("Enter your equation (e.g., x^2 - 4 = 0):")
    if st.button("Solve 🧮"):
        try:
            eq = sp.sympify(latex_input.replace("^","**"))
            sol = sp.solve(eq, dict=True)
            if sol:
                st.success("✅ Solution found!")
                for s in sol:
                    st.latex(f"x = {sp.latex(list(s.values())[0])}")
            else:
                st.warning("⚠️ No solution found. Try another equation.")
        except Exception as e:
            st.error(f"❌ Invalid equation: {e}")

# ------------------ Scientific Calculator (Desmos-like) ------------------
elif mode == "Scientific Calculator (Desmos-like)":
    st.subheader("📊 Scientific Calculator (Desmos Style)")
    st.info("Enter one or multiple functions separated by commas (e.g., x^2, sin(x), cos(x))")
    func_input = st.text_input("Enter function(s):")
    plot_range = st.slider("Select X range", -20, 20, (-10,10))
    
    if st.button("Plot Functions 📈"):
        try:
            funcs = [sp.sympify(f.replace("^","**")) for f in func_input.split(",")]
            x = sp.symbols('x')
            x_vals = np.linspace(plot_range[0], plot_range[1], 400)
            plt.figure(figsize=(8,5))
            for f in funcs:
                y_vals = [f.subs(x,val) for val in x_vals]
                plt.plot(x_vals, y_vals, label=f"{f}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("📈 Graph of your functions")
            plt.legend()
            st.pyplot(plt)
            st.success("🎉 Plot generated successfully!")
        except Exception as e:
            st.error(f"❌ Invalid function input: {e}")

# ------------------ Google Scholar Search ------------------
elif mode == "Google Scholar Search":
    st.subheader("🔬 Google Scholar Search")
    keyword = st.text_input("Enter research topic:")
    if st.button("Search Scholar 🔍"):
        st.info("Fetching top research papers... ⏳")
        query = f"{keyword} site:scholar.google.com"
        results = []
        try:
            for url in search(query, num_results=5):
                results.append(url)
        except:
            st.error("❌ Error searching Scholar.")

        if results:
            st.success(f"Found {len(results)} papers:")
            for r in results:
                try:
                    r_page = requests.get(r, timeout=3)
                    soup = BeautifulSoup(r_page.text, "html.parser")
                    title = soup.title.string if soup.title else r
                    snippet = " ".join([p.get_text()[:150]+"..." for p in soup.find_all('p')[:2]])
                    st.markdown(f"📄 **{title}**\n{snippet}\n🔗 [Read More]({r})")
                except:
                    st.markdown(f"📄 {r}")
        else:
            st.warning("⚠️ No papers found. Try a different keyword.")

# ------------------ 2D & 3D Graph Generator ------------------
elif mode == "2D & 3D Graph Generator":
    st.subheader("📊 2D & 3D Graph Generator")
    raw_input = st.text_input("Enter function (2D: x only, 3D: z=f(x,y), e.g., sin(x)*cos(y)):")
    func_input = raw_input.replace("^","**").replace("X","x").replace("Y","y")
    graph_type = st.radio("Graph Type:", ["2D", "3D"])
    if st.button("Plot Graph"):
        try:
            if graph_type == "2D":
                x = sp.symbols('x')
                func = sp.sympify(func_input)
                x_vals = np.linspace(-10, 10, 400)
                y_vals = [func.subs(x,i) for i in x_vals]
                plt.plot(x_vals, y_vals)
                plt.xlabel("x")
                plt.ylabel("y")
                plt.title(f"Graph of {func_input} 📈")
                st.pyplot(plt)
            else:
                x, y = sp.symbols('x y')
                func = sp.sympify(func_input)
                X = np.linspace(-5,5,100)
                Y = np.linspace(-5,5,100)
                X, Y = np.meshgrid(X, Y)
                Z_func = sp.lambdify((x,y), func, "numpy")
                Z = Z_func(X,Y)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(X, Y, Z, cmap='viridis')
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                plt.title(f"3D Graph of {func_input} 📊")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Invalid equation: {e}")
