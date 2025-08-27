# app.py - ERIK v6 Full (updated quiz + long/short answers + detailed math solve)
import streamlit as st
import requests
from bs4 import BeautifulSoup
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import fitz  # PyMuPDF
import docx
from googlesearch import search
import random
import re
from html import unescape

# ------------------ ERIK v6 Full ------------------
st.set_page_config(page_title="ERIK v6 - AI Academic Assistant", layout="wide")

# ------------------ Introduction ------------------
st.title("🧠 ERIK v6 - Exceptional Resources & Intelligence Kernel")
st.markdown("""
**Welcome to ERIK v6!**  
ERIK is an AI-powered academic assistant designed to help students and researchers with **math, graphing, research, quizzes** and more.  
✨ **Created and developed by Sabid Uddin Nahian**.  
Use the sidebar to explore the features! 🚀
""")

# ------------------ Sidebar ------------------
st.sidebar.header("Features")
mode = st.sidebar.radio("Choose a feature:", [
    "Ask Question",
    "Quiz Generator",
    "PDF/Text Analyzer",
    "LaTeX Math Solver (detailed)",
    "Scientific Calculator (Desmos-like)",
    "Google Scholar Search",
    "2D & 3D Graph Generator"
])

# ------------------ Utilities ------------------
def fetch_text_from_url(url, max_paragraphs=5):
    try:
        r = requests.get(url, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        paras = soup.find_all('p')
        text = " ".join([unescape(p.get_text().strip()) for p in paras[:max_paragraphs]])
        return text
    except Exception:
        return ""

def extract_sentences(text):
    # naive sentence splitter
    sents = re.split(r'(?<=[\.\?\!])\s+', text)
    sents = [s.strip() for s in sents if len(s.strip())>20]
    return sents

def choose_targets(sentences):
    # pick sentences that contain numbers or capitalized entities as targets
    targets = []
    for s in sentences:
        if re.search(r'\d', s):
            targets.append(("numeric", s))
        elif re.search(r'\b[A-Z][a-z]{2,}\b', s):
            targets.append(("entity", s))
        else:
            targets.append(("text", s))
    return targets

def generate_mcq_from_sentence(sent, kind="numeric"):
    """
    Heuristic MCQ generation:
    - If numeric: find first number and make distractors ± or scaled
    - If entity: mask a proper noun and create distractors from other proper nouns in text
    - Else: mask a keyword (longest noun-ish word)
    Returns: (question_text, options_list, correct_index, solution_text)
    """
    # numeric case
    nums = re.findall(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?', sent)
    if nums:
        target = nums[0]
        try:
            val = float(target)
            # create distractors
            d1 = val + 1
            d2 = val - 1
            d3 = val * 1.5 if abs(val) > 1e-6 else val + 2
            opts = [str(val), str(round(d1,3)), str(round(d2,3)), str(round(d3,3))]
            random.shuffle(opts)
            correct = opts.index(str(val))
            q = re.sub(re.escape(target), "_____", sent, count=1)
            sol = f"Answer: {val}"
            return q, opts, correct, sol
        except:
            pass

    # entity case: choose a capitalized token as answer
    caps = re.findall(r'\b([A-Z][a-z]{2,})\b', sent)
    if caps:
        ans = caps[0]
        # create distractors by picking other caps from sentence or by simple perturbation
        others = caps[1:] + [w for w in caps if w!=ans]
        distractors = []
        for o in others[:3]:
            if o != ans:
                distractors.append(o)
        # if not enough, make synthetic ones by appending letters
        while len(distractors) < 3:
            distractors.append(ans + random.choice(["a","o","i","er"]))
        opts = [ans] + distractors[:3]
        random.shuffle(opts)
        correct = opts.index(ans)
        q = re.sub(r'\b' + re.escape(ans) + r'\b', "_____", sent, count=1)
        sol = f"Answer: {ans}"
        return q, opts, correct, sol

    # fallback: mask a long word
    words = [w for w in re.findall(r'\w+', sent) if len(w)>4]
    if words:
        ans = words[0]
        distractors = []
        # pick similar length words or alter the correct answer
        for _ in range(3):
            distractors.append(ans[:max(1,len(ans)-1)] + random.choice(["a","o","x","z"]))
        opts = [ans] + distractors
        random.shuffle(opts)
        correct = opts.index(ans)
        q = re.sub(r'\b' + re.escape(ans) + r'\b', "_____", sent, count=1)
        sol = f"Answer: {ans}"
        return q, opts, correct, sol

    # as ultimate fallback produce an SAQ style
    q = sent if len(sent)<120 else sent[:120] + "..."
    return q, ["Option A","Option B","Option C","Option D"], 0, "See text"

def generate_quiz_from_text(content, num_q=5):
    sentences = extract_sentences(content)
    if not sentences:
        return []
    targets = choose_targets(sentences)
    quiz_items = []
    used = set()
    i=0
    for kind, sent in targets:
        if i>=num_q:
            break
        if sent in used: 
            continue
        used.add(sent)
        # decide type randomly but ensure a mix
        qtype = random.choices(["MCQ","SAQ","CQ"], weights=[0.6,0.2,0.2])[0]
        if qtype == "MCQ":
            q, opts, corr, sol = generate_mcq_from_sentence(sent, kind)
            quiz_items.append({"type":"MCQ","question":q,"options":opts,"answer_index":corr,"solution":sol})
        elif qtype == "SAQ":
            # generate short answer by extracting a short phrase (numbers or capitalized word)
            nums = re.findall(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?', sent)
            if nums:
                ans = nums[0]
            else:
                caps = re.findall(r'\b([A-Z][a-z]{2,})\b', sent)
                ans = caps[0] if caps else sent.split()[0]
            quiz_items.append({"type":"SAQ","question": "Short Answer: " + sent,"answer":ans,"solution":f"Answer: {ans}"})
        else:
            # CQ: create a comprehension prompt + model answer from context
            # use surrounding sentences as brief model answer
            idx = sentences.index(sent)
            context = " ".join(sentences[max(0, idx-1): min(len(sentences), idx+2)])
            qtext = "Explain / discuss: " + sent
            model = context
            quiz_items.append({"type":"CQ","question":qtext,"model_answer":model})
        i+=1
    return quiz_items

# ------------------ Ask Question (Multilingual) with short/long answer ------------------
if mode == "Ask Question":
    st.subheader("🌐 Ask Question (English, Bangla, German & Mandarin)")
    language = st.selectbox("Select language:", ["English", "Bangla", "German", "Mandarin"])
    format_choice = st.selectbox("Answer format:", ["Short", "Long"])
    query = st.text_input("Enter your question:")
    if st.button("Search & Answer 🔍"):
        st.info("Searching the web... ⏳")
        lang_code = {"English":"en", "Bangla":"bn", "German":"de", "Mandarin":"zh-CN"}[language]
        results = []
        try:
            for url in search(query, num_results=5, lang=lang_code):
                results.append(url)
        except Exception:
            st.error("❌ Error searching the web.")
        answer = ""
        for link in results:
            answer += fetch_text_from_url(link, max_paragraphs=3) + " "
        if answer.strip():
            if format_choice == "Short":
                # short: first 1-2 sentences
                sents = extract_sentences(answer)
                short_ans = sents[0] if sents else answer[:300]
                st.markdown("💡 **Quick answer:**")
                st.write(f"🤖 {short_ans.strip()}")
            else:
                # long: give a summarized paragraph (naive summarization: first 5 sentences)
                sents = extract_sentences(answer)
                long_ans = " ".join(sents[:6]) if sents else answer
                st.markdown("💡 **Detailed answer:**")
                st.write(f"🤖 {long_ans.strip()}")
            st.markdown("🔗 **Top sources:**")
            for r in results:
                st.write(f"- {r}")
        else:
            st.warning("⚠️ Hmm… I couldn't find an answer. Try asking differently!")

# ------------------ Quiz Generator (Google + PDF) ------------------
elif mode == "Quiz Generator":
    st.subheader("📝 Smart Quiz Generator (MCQ / SAQ / CQ)")
    source_type = st.radio("Select source type:", ["Google Search", "PDF/Text Upload"])
    num_q = st.number_input("Number of questions to generate:", min_value=1, max_value=20, value=5)
    if source_type == "Google Search":
        topic = st.text_input("Enter topic for quiz generation:")
        if st.button("Generate Quiz from Google"):
            st.info("Searching Google and creating quiz...")
            content = ""
            try:
                for url in search(topic, num_results=6):
                    content += fetch_text_from_url(url, max_paragraphs=4) + " "
                quiz_items = generate_quiz_from_text(content, num_q)
                if not quiz_items:
                    st.warning("No useful text found to generate questions.")
                else:
                    for idx, item in enumerate(quiz_items, start=1):
                        if item["type"] == "MCQ":
                            st.write(f"Q{idx} (MCQ): {item['question']}")
                            for i,opt in enumerate(item["options"]):
                                st.write(f"   {chr(97+i)}) {opt}")
                            with st.expander("Show solution & explanation"):
                                st.write(item.get("solution",""))
                                # attempt automatic solve if numeric present
                                if re.search(r'\d', item.get("solution","")):
                                    st.write("Auto-solved answer:", item.get("solution"))
                        elif item["type"] == "SAQ":
                            st.write(f"Q{idx} (SAQ): {item['question']}")
                            with st.expander("Answer"):
                                st.write(item.get("answer",""))
                                st.write(item.get("solution",""))
                        else:
                            st.write(f"Q{idx} (CQ): {item['question']}")
                            with st.expander("Model answer / rubric"):
                                st.write(item.get("model_answer",""))
            except Exception as e:
                st.error(f"❌ Error generating quiz: {e}")

    else:  # PDF/Text Upload
        uploaded_file = st.file_uploader("Upload PDF / DOCX / TXT file", type=['pdf','txt','docx'])
        if uploaded_file and st.button("Generate Quiz from PDF/Text"):
            text = ""
            try:
                if uploaded_file.type == "application/pdf":
                    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                    for page in doc:
                        text += page.get_text()
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    document = docx.Document(uploaded_file)
                    for para in document.paragraphs:
                        text += para.text + "\n"
                else:
                    text = str(uploaded_file.read(), "utf-8")
                quiz_items = generate_quiz_from_text(text, num_q)
                if not quiz_items:
                    st.warning("Not enough text to generate questions.")
                else:
                    for idx, item in enumerate(quiz_items, start=1):
                        if item["type"] == "MCQ":
                            st.write(f"Q{idx} (MCQ): {item['question']}")
                            for i,opt in enumerate(item["options"]):
                                st.write(f"   {chr(97+i)}) {opt}")
                            with st.expander("Show solution & explanation"):
                                st.write(item.get("solution",""))
                        elif item["type"] == "SAQ":
                            st.write(f"Q{idx} (SAQ): {item['question']}")
                            with st.expander("Answer"):
                                st.write(item.get("answer",""))
                                st.write(item.get("solution",""))
                        else:
                            st.write(f"Q{idx} (CQ): {item['question']}")
                            with st.expander("Model answer / rubric"):
                                st.write(item.get("model_answer",""))
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
        st.text_area("Extracted Text", text, height=400)

# ------------------ LaTeX Math Solver (detailed) ------------------
elif mode == "LaTeX Math Solver (detailed)":
    st.subheader("✍️ LaTeX Math Solver — Detailed Steps")
    latex_input = st.text_input("Enter expression/equation. For equation add '=0' (e.g., x^2 - 4 = 0) or expression like 'integrate sin(x) dx':")
    if st.button("Solve with steps"):
        try:
            # parse equality
            expr = latex_input.replace("^","**")
            # If contains '=' treat as equation
            if "=" in expr:
                left, right = expr.split("=",1)
                eq_expr = sp.sympify(left) - sp.sympify(right)
                st.markdown("**1) Original equation (converted):**")
                st.latex(sp.latex(sp.Eq(sp.sympify(left), sp.sympify(right))))
                st.markdown("**2) Simplify:**")
                simp = sp.simplify(eq_expr)
                st.latex(sp.latex(simp) + " = 0")
                st.markdown("**3) Factor (if possible):**")
                fact = sp.factor(simp)
                st.latex(sp.latex(fact) + " = 0")
                st.markdown("**4) Solve:**")
                sols = sp.solve(simp, dict=False)
                if sols:
                    for s in sols:
                        st.latex("x = " + sp.latex(sp.nsimplify(s)))
                else:
                    st.write("No symbolic solutions found.")
                st.markdown("**5) Check (substitute solution back):**")
                for s in sols:
                    check = sp.simplify(simp.subs(sp.symbols('x'), s))
                    st.write(f"Substituting x={s} → {check}")
            else:
                # treat as expression: try simplify, derivative, integral
                e = sp.sympify(expr)
                st.markdown("**Expression:**")
                st.latex(sp.latex(e))
                st.markdown("**Simplified:**")
                st.latex(sp.latex(sp.simplify(e)))
                st.markdown("**Factorized (if any):**")
                st.latex(sp.latex(sp.factor(e)))
                st.markdown("**Derivative (w.r.t x):**")
                x = sp.symbols('x')
                try:
                    der = sp.diff(e, x)
                    st.latex(sp.latex(der))
                except Exception:
                    st.write("Derivative not applicable / symbolic failure.")
                st.markdown("**Indefinite integral (w.r.t x):**")
                try:
                    inte = sp.integrate(e, x)
                    st.latex(sp.latex(inte))
                except Exception:
                    st.write("Integral not found symbolically.")
        except Exception as e:
            st.error(f"❌ Invalid input or cannot solve symbolically: {e}")

# ------------------ Scientific Calculator (Desmos-like) ------------------
elif mode == "Scientific Calculator (Desmos-like)":
    st.subheader("📊 Scientific Calculator (Desmos Style)")
    st.info("Enter one or multiple functions separated by commas (e.g., x^2, sin(x), cos(x))")
    func_input = st.text_input("Enter function(s):")
    plot_range = st.slider("Select X range", -20, 20, (-10,10))
    realtime = st.checkbox("Show grid & axes", value=True)
    if st.button("Plot Functions 📈"):
        try:
            funcs = [sp.sympify(f.replace("^","**")) for f in func_input.split(",") if f.strip()]
            if not funcs:
                st.warning("Please enter at least one function.")
            else:
                x = sp.symbols('x')
                x_vals = np.linspace(plot_range[0], plot_range[1], 800)
                fig, ax = plt.subplots(figsize=(8,5))
                for f in funcs:
                    # lambdify to numpy for speed
                    f_np = sp.lambdify((x,), f, "numpy")
                    try:
                        y_vals = f_np(x_vals)
                    except Exception:
                        # fallback evaluate pointwise
                        y_vals = np.array([float(f.subs(x,val)) for val in x_vals], dtype=float)
                    ax.plot(x_vals, y_vals, label=str(f))
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_title("📈 Graph of your functions")
                ax.legend()
                if realtime:
                    ax.grid(True)
                    ax.axhline(0, color='black', linewidth=0.5)
                    ax.axvline(0, color='black', linewidth=0.5)
                st.pyplot(fig)
                st.success("🎉 Plot generated successfully!")
        except Exception as e:
            st.error(f"❌ Invalid function input or plotting error: {e}")

# ------------------ Google Scholar Search (clean professional) ------------------
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
        except Exception:
            st.error("❌ Error searching Scholar.")
        if results:
            st.success(f"Found {len(results)} papers:")
            for r in results:
                try:
                    r_page = requests.get(r, timeout=4)
                    soup = BeautifulSoup(r_page.text, "html.parser")
                    title = (soup.title.string.strip() if soup.title else r)
                    para_elems = soup.find_all('p')
                    snippet = " ".join([unescape(p.get_text()[:150]).strip() + "..." for p in para_elems[:2]]) if para_elems else ""
                    st.markdown(f"**{title}**")
                    if snippet:
                        st.write(snippet)
                    st.write(f"[Read more]({r})")
                    st.divider()
                except Exception:
                    st.write(r)
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
                f_np = sp.lambdify((x,), func, "numpy")
                y_vals = f_np(x_vals)
                fig, ax = plt.subplots(figsize=(7,4))
                ax.plot(x_vals, y_vals)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_title(f"Graph of {func_input} 📈")
                st.pyplot(fig)
            else:
                x, y = sp.symbols('x y')
                func = sp.sympify(func_input)
                X = np.linspace(-5,5,120)
                Y = np.linspace(-5,5,120)
                Xg, Yg = np.meshgrid(X, Y)
                Z_func = sp.lambdify((x,y), func, "numpy")
                Z = Z_func(Xg, Yg)
                fig = plt.figure(figsize=(8,6))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(Xg, Yg, Z, cmap='viridis', linewidth=0, antialiased=True)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_title(f"3D Graph of {func_input} 📊")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"❌ Invalid equation or plotting error: {e}")

# ------------------ End ------------------
st.markdown("---")
st.markdown("ERIK v6 — built with ❤️ by **Sabid bhai**. If something breaks, send me the traceback and I'll patch it.")
