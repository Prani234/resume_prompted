import os
import re
import json
import requests
import streamlit as st
import PyPDF2
import docx
from datetime import datetime
from pdf2image import convert_from_path
import pytesseract
from dotenv import load_dotenv
from TTS.api import TTS
import dateparser

# LangChain / Groq imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# -------------------- ENV SETUP --------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
GROQ_API_BASE = "https://api.groq.com/openai/v1"

UPLOAD_PATH = "__DATA__"
os.makedirs(UPLOAD_PATH, exist_ok=True)

st.set_page_config(page_title="AI Resume Assistant", layout="wide")
st.title("🤖 AI Resume Analyzer")

# =====================================================
# TEXT EXTRACTION UTILS
# =====================================================
def extract_text_and_links(file_path, file_type="pdf"):
    text, links = "", []
    try:
        if file_type == "pdf":
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += (page.extract_text() or "") + "\n"
                    if "/Annots" in page:
                        for annot in page["/Annots"]:
                            obj = annot.get_object()
                            if "/A" in obj and "/URI" in obj["/A"]:
                                links.append(obj["/A"]["/URI"])
        elif file_type == "docx":
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
            for rel in doc.part.rels.values():
                if "hyperlink" in rel.reltype:
                    links.append(rel.target_ref)
        elif file_type == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        return {"text": text.strip(), "links": sorted(set(links))}
    except Exception as e:
        return {"text": f"__READ_ERROR__:{e}", "links": []}


# =====================================================
# OCR VALIDATION
# =====================================================
def is_image_based_resume(pdf_path, extracted_text, text_threshold=300):
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        total_pages = len(reader.pages)
        if total_pages == 0:
            return True

        sample_pages = [0, total_pages // 2, total_pages - 1]
        images = convert_from_path(pdf_path, first_page=1, last_page=total_pages)
        for i, p in enumerate(sample_pages):
            page_text = reader.pages[p].extract_text() or ""
            ocr_text = pytesseract.image_to_string(images[p])
            if len(ocr_text.strip()) > len(page_text.strip()) + 10:
                return True
        if len(extracted_text.strip()) < text_threshold:
            return True
        return False
    except Exception:
        return True


# =====================================================
# GROQ ATS EXTRACTOR
# =====================================================
def ats_extractor(resume_text, structured_links):
    prompt = f"""
    You are a highly accurate resume parser.
    Parse the resume below into structured JSON.

    Include fields:
    classification (Fresher/Experienced),
    Name, Email, Phone, Education, Experience, Internships, Projects,
    Technical Skills, Soft Skills, Certifications, Languages, Hobbies, Summary,
    CTC details, Work Preferences, Profile Links, etc.

    Use empty strings/lists for missing fields.
    Return **only valid JSON**, no commentary.

    Structured Links:
    {json.dumps(structured_links, indent=2)}

    Resume Text:
    {resume_text}
    """

    url = f"{GROQ_API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 2500
    }

    try:
        res = requests.post(url, headers=headers, json=data)
        content = res.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{[\s\S]*\}", content)
        return json.loads(match.group(0)) if match else {}
    except Exception as e:
        st.error(f"API error: {e}")
        return {}


# =====================================================
# LINK CLASSIFIER
# =====================================================
def classify_links(links):
    patterns = {
        "LinkedIn": "linkedin\\.com",
        "GitHub": "github\\.com",
        "Behance": "behance\\.net"
    }
    classified = {"profile_links": {}, "certificate_links": [], "other_links": []}
    for link in links:
        found = False
        for key, pat in patterns.items():
            if re.search(pat, link, re.I):
                classified["profile_links"][key] = link
                found = True
        if not found:
            classified["other_links"].append(link)
    return classified


# =====================================================
# SELF INTRO GENERATOR
# =====================================================
def generate_intro(parsed_resume):
    prompt = f"""
    Based on the JSON resume below, create a 45-second self-introduction script 
    suitable for interviews. Include education, skills, projects/experience, and career goals.
    Make it professional, fluent, and confident.

    Resume JSON:
    {json.dumps(parsed_resume, indent=2)}
    """

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 400
    }

    try:
        res = requests.post(f"{GROQ_API_BASE}/chat/completions", headers=headers, json=data)
        return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error generating intro: {e}"


# =====================================================
# TEXT TO SPEECH
# =====================================================
def text_to_speech(script_text):
    audio_path = os.path.join(UPLOAD_PATH, "intro.wav")
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
    tts.tts_to_file(text=script_text, file_path=audio_path)
    return audio_path


# =====================================================
# MAIN STREAMLIT FLOW
# =====================================================
st.header("📂 Step 1: Upload Resume")
uploaded = st.file_uploader("Upload PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])

if uploaded:
    # Save file
    file_ext = uploaded.name.split(".")[-1].lower()
    file_path = os.path.join(UPLOAD_PATH, f"resume.{file_ext}")
    with open(file_path, "wb") as f:
        f.write(uploaded.read())

    # Extract text and validate
    st.info("Extracting text...")
    data_obj = extract_text_and_links(file_path, file_type=file_ext)

    if data_obj["text"].startswith("__READ_ERROR__"):
        st.error("Error reading file.")
        st.stop()

    if file_ext == "pdf" and is_image_based_resume(file_path, data_obj["text"]):
        st.error("❌ Image-based/scanned resume detected. Please upload a text-based file.")
        st.stop()

    st.success("✅ Resume text extracted successfully.")

    # Parse resume to JSON
    structured_links = classify_links(data_obj["links"])
    ats_json = ats_extractor(data_obj["text"], structured_links)

    if not ats_json:
        st.error("Resume parsing failed.")
        st.stop()

    st.subheader("📋 Extracted Resume Data")
    st.json(ats_json)

    # Store parsed data
    # st.session_state.resume_data = ats_json
    # resume_json_str = json.dumps(st.session_state.resume_data, indent=2)
    # st.write("DEBUG:", resume_json_str)

    # ----------------- STEP 2: CHATBOT -----------------
    st.header("💬 Step 2: Fill Missing Details")

    def find_missing_fields(resume_json):
        missing = []
        for k, v in resume_json.items():
            if isinstance(v, (list, dict)) and not v:
                missing.append(k)
            elif isinstance(v, str) and not v.strip():
                missing.append(k)
            elif v is None:
                missing.append(k)
        return missing


    missing = find_missing_fields(st.session_state.resume_data)
    if not missing:
        st.success("✅ All fields already filled!")
    else:
        groq_api_key = os.getenv("GROQ_API_KEY")
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0.3)

        system_prompt = """
        You are a friendly and professional resume-building assistant (DO NOT mention this in chat).

        Your job:
        - Help the user fill out only the **missing or empty fields** in their resume step-by-step.
        - Use the provided JSON to identify which fields are empty.
        - If a field already has a value, **do not ask about it again**.
        - Always ask **one clear and specific question** about the next missing field.
        - Keep responses short, conversational, and relevant to resume completion.
        - Never show or mention the JSON structure to the user.
        - Once all fields are filled, say: "Great! Your resume is complete."
        - Do not answer unrelated questions.

        Here is the current resume data (with some fields possibly empty):
        {resume_json}
        """


        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt + "\nCurrent JSON:\n{resume_json}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        msgs = StreamlitChatMessageHistory(key="chat_history")
        chain = prompt | llm
        with_history = RunnableWithMessageHistory(
            chain, lambda _: msgs, input_messages_key="input", history_messages_key="chat_history"
        )

        if "messages" not in st.session_state:
            resume_json_str = json.dumps(st.session_state.resume_data, indent=2)
            try:
                first_response = llm.invoke([
                    {"role": "system", "content": system_prompt.format(resume_json=resume_json_str)},
                    {"role": "user", "content": "Start with a greeting and ask about the first missing field."}
                ])
                first_question = first_response.content.strip()
            except Exception:
                first_question = "Hi! Let's start by confirming your full name."

            st.session_state.messages = [{"role": "assistant", "content": first_question}]

        chat_box = st.container(height=500, border=True)
        with chat_box:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

        if user_input := st.chat_input("Your response..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            missing = find_missing_fields(st.session_state.resume_data)
            if missing:
                st.session_state.resume_data[missing[0]] = user_input

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = with_history.invoke(
                        {
                            "input": user_input,
                            "resume_json": json.dumps(st.session_state.resume_data, indent=2)
                        },
                        config={"configurable": {"session_id": "resume_bot"}}
                    )
                    reply = response.content
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})

            missing = find_missing_fields(st.session_state.resume_data)
            if not missing:
                st.success("✅ Resume is complete!")
                with st.expander("📄 Final Resume JSON"):
                    st.json(st.session_state.resume_data)

                # ----------------- STEP 3: INTRO -----------------
                st.header("🎙 Step 3: Generate Self-Introduction")
                if st.button("Generate Introduction"):
                    intro = generate_intro(st.session_state.resume_data)
                    st.subheader("🗣 Self-Introduction Script")
                    st.write(intro)
                    try:
                        audio_file = text_to_speech(intro)
                        st.audio(audio_file)
                    except Exception as e:
                        st.error(f"TTS failed: {e}")
