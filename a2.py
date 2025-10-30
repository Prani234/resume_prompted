import os
import streamlit as st
import PyPDF2
import json
import requests
import re
from dotenv import load_dotenv
from TTS.api import TTS
from datetime import datetime
import dateparser

from pdf2image import convert_from_path
import pytesseract
import docx

# ---------------- Load environment variables ----------------
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
GROQ_API_BASE = "https://api.groq.com/openai/v1"

# Directory for uploads
UPLOAD_PATH = "__DATA__"
os.makedirs(UPLOAD_PATH, exist_ok=True)

# ---------------- PDF / DOCX / TXT Extraction ----------------
def extract_text_and_links(file_path, file_type="pdf"):
    """
    Extract text and hyperlinks from PDF, DOCX, or TXT files.
    Returns a dictionary with 'text' and 'links' keys.
    """
    text = ""
    links = []

    try:
        if file_type == "pdf":
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
                    # Extract links
                    if "/Annots" in page:
                        for annot in page["/Annots"]:
                            obj = annot.get_object()
                            if "/A" in obj and "/URI" in obj["/A"]:
                                links.append(obj["/A"]["/URI"])

        elif file_type == "docx":
            # Word (.docx) file extraction
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
            # Optional: extract hyperlinks from runs
            for rel in doc.part.rels.values():
                if "hyperlink" in rel.reltype:
                    links.append(rel.target_ref)

        elif file_type == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        return {"text": text.strip(), "links": sorted(set(links))}

    except Exception as e:
        return {"text": f"__READ_ERROR__:{str(e)}", "links": []}


# ---------------- OCR-Based Image Resume Detection ----------------
def is_image_based_resume(pdf_path, extracted_text, text_threshold=300, sample_pages=3):
    """
    Strict detection: If *any* page contains image-based text, mark as invalid.
    Detects even partial image-based headers or text regions.
    Returns True (invalid) if OCR detects any image text.
    """

    try:
        reader = PyPDF2.PdfReader(pdf_path)
        total_pages = len(reader.pages)

        if total_pages == 0:
            return True  # Empty PDF = invalid

        # Sample first, middle, last pages (efficient)
        if total_pages <= sample_pages:
            pages_to_check = list(range(total_pages))
        else:
            mid = total_pages // 2
            pages_to_check = [0, mid, total_pages - 1]

        # Convert only those sampled pages into images
        images = convert_from_path(pdf_path, first_page=min(pages_to_check)+1, last_page=max(pages_to_check)+1)

        for idx, page_num in enumerate(pages_to_check):
            page_text = reader.pages[page_num].extract_text() or ""
            ocr_text = pytesseract.image_to_string(images[idx])

            # Strict rule:
            # If OCR finds text that's not in PyPDF2 extraction — even a little — mark invalid
            if len(ocr_text.strip()) > len(page_text.strip()) + 10:
                return True  # Partially or fully image-based

        # Also reject if total extracted text is too small
        if len(extracted_text.strip()) < text_threshold:
            return True

        return False  # Valid, text-based resume

    except Exception as e:
        print("OCR check failed:", e)
        return True

# ---------------- Link Classification ----------------
def classify_links(links):
    profile_patterns = {
        "LinkedIn": r'linkedin\.com',
        "GitHub": r'github\.com$',  # GitHub profile only
        "Behance": r'behance\.net',
        "PersonalWebsite": r'personalwebsite\.com'
    }
    certificate_patterns = [
        r'certificates\.edu',
        r'docsend\.com',
        r'publication\.org'
    ]

    classified = {"profile_links": {}, "certificate_links": [], "other_links": []}

    for link in links:
        matched = False
        for domain, pattern in profile_patterns.items():
            if re.search(pattern, link, re.I):
                classified["profile_links"][domain] = link
                matched = True
                break

        if not matched:
            if any(re.search(p, link, re.I) for p in certificate_patterns):
                classified["certificate_links"].append(link)
            else:
                classified["other_links"].append(link)

    classified["certificate_links"] = sorted(set(classified["certificate_links"]))
    classified["other_links"] = sorted(set(classified["other_links"]))

    return classified

# ---------------- Duration Parsing ----------------
def parse_duration(period_text):
    period_text = period_text.replace("–", "-").replace("to", "-").strip()
    parts = [p.strip() for p in period_text.split("-")]
    if len(parts) != 2:
        return ""
    start_str, end_str = parts
    end_str_clean = end_str.lower().replace("\u00A0", "").strip()
    start_date = dateparser.parse(start_str)
    if end_str_clean in ["present", "current"]:
        end_date = datetime.today()
    else:
        end_date = dateparser.parse(end_str)
    if not start_date or not end_date:
        return ""
    months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    years = months // 12
    rem_months = months % 12
    if years > 0 and rem_months > 0:
        return f"{years} yr {rem_months} months"
    elif years > 0:
        return f"{years} yr"
    else:
        return f"{rem_months} months"

def add_durations_to_experiences(experience_list):
    for exp in experience_list:
        period_text = exp.get("Period", "")
        exp["Duration"] = parse_duration(period_text)
    return experience_list

# ---------------- Project Links Assignment ----------------
def assign_project_links(resume_text, project_sections, all_links):
    projects_with_links = []
    for project in project_sections:
        description = project.get("Description", "")
        project_links = [link for link in all_links if link in description]
        project["project_links"] = project_links
        projects_with_links.append(project)
    return projects_with_links

# ---------------- ATS Extractor ----------------
def ats_extractor(resume_text, structured_links, experiences=[], internships=[]):
    experiences = add_durations_to_experiences(experiences)
    internships = add_durations_to_experiences(internships)

    prompt = (
        "You are an AI bot parsing resumes with high accuracy (target 95%). "
        "First, determine if the resume belongs to a **Fresher** (no professional work experience beyond internships, "
        "recent graduation within 2 years, focus on education/projects) or an **Experienced professional** "
        "(mentions job roles, companies, and total experience > 1 year). Use these criteria strictly:\n"
        "- Fresher: Internship duration < 6 months, no full-time roles, recent education (e.g., 2023-2025).\n"
        "- Experienced: Full-time roles with total experience > 1 year, or senior roles.\n"
        "Return a JSON object with a mandatory 'classification' field ('Fresher' or 'Experienced') and extract the following fields based on the category. "
        "Ensure all fields are included, using empty strings ('') for unavailable data. Output only valid JSON, no additional text.\n\n"

        "Important Classification Refinement Rules:\n"
        "- If graduation year is within 2023–2025, assume the candidate is likely a Fresher unless there is clear, multi-year full-time work after graduation.\n"
        "- Treat 'Live Project', 'Internship', 'Academic Project', or 'Capstone Project' as internship experience — never as full-time work.\n"
        "- If the experience period overlaps with college years, treat it as internship or academic work, not full-time.\n"
        "- Do not assume the candidate is Experienced simply because there is a 'Work Experience' section.\n"
        "- If total full-time experience duration is less than 12 months, classify as 'Fresher'.\n"
        "- Prefer 'Fresher' classification when ambiguous — do not overestimate experience.\n"

        "Return a JSON object with a mandatory 'classification' field ('Fresher' or 'Experienced') and extract the following fields based on the category. "
        "Ensure all fields are included, using empty strings ('') for unavailable data. Output only valid JSON, no additional text.\n\n"


        "- For Experienced candidates, `experience.jobs` should **only include full-time jobs**."
        "- Internships must be listed **only in the `internships` field**, even if after graduation."
        "- Do not duplicate internships in `experience.jobs`."
        "- If the candidate has no full-time experience, set 'total_experience' to  null."
          "Do not guess or fill values."
        "- If any field is missing, return it as an empty string, empty list, or null."


        "Additionally, extract a dedicated field:\n"
        "'Technical skills' → a combined list of *every skill mentioned anywhere* "
        "(Skills section, Projects, Experience, Certifications, Summary, etc).\n\n"

        """... For fields like `Description` and `Responsibilities` in Experience, Projects, and Internships, 
            summarize them into 3–4 concise sentences that preserve all the important technical and contextual details. 
            Do not drop key technologies, roles, or outcomes. The goal is to condense long paragraphs into shorter 
            summaries without losing meaning."""

        """- For the 'Summary' field, condense the candidate’s profile summary into 3–5 sentences. 
             Retain core skills, domain expertise, career goals, and unique strengths. 
             Avoid generic filler text. Ensure the summary is professional, concise, and impactful."""



        "If Fresher, extract:\n"
        "1. Name\n2. Email\n3. Phone number\n4. Profile links (LinkedIn, GitHub, etc)\n"
        "5. Education details (list of {Degree(include specialization if mentioned), College, Year, Grades})\n"
        "- Do not separate specialization into a new field. Include it in Degree.\n"
        "6. Technical skills (include skills mentioned in Skills section, job descriptions, project descriptions,Certifications and summary)\n"
        "7. Soft skills (list)\n8. Projects (list of {Title, Description,Technologies used, Responsibilities,project links if provided})\n"
        "9. Certifications (list)\n10. Languages known (list)\n11. Hobbies/extra-curriculars\n"
        "12. Current location\n13. Expected CTC\n14. Work/location preference\n15. work mode\n"
        "16. Internship details (list of {Company,Role,Period, Duration,Technologies used in each role, Responsibilities})\n17. Summary\n\n"

        "If Experienced, extract:\n"
        "1. Name\n2. Email\n3. Phone number\n4. Profile links (LinkedIn, GitHub, etc)\n"
        "5. Technical skills (include skills mentioned in Skills section, job descriptions, project descriptions,Certifications and summary)\n6. Soft skills (list)\n"
        "7. Total Experience\n"  
        "8. Experience details (list of {Company, Role,Period, Duration, Technologies used, Responsibilities})\n"
        "    - Only full-time professional roles. Do not include internships here.\n"
        "9. Internship details (list of {Company, Role,Period,Duration,Technologies used, Responsibilities})\n"
        "    - Only internships. Do not duplicate in Experience details.\n"
        "10. Projects (list of {Title, Description,Technologies used, Responsibilities,project links if provided})\n"
        "11. Certifications (list)\n"
        "12. Education details (list of {Degree(include specialization if mentioned), College, Year, Grades})\n"
             "- Do not separate specialization into a new field. Include it in Degree.\n"
        "13. Current CTC\n14. Expected CTC\n"
        "15. Notice period\n16. Work mode\n17. Preferred location\n"
        "18. Summary\n19. Languages known (list)\n"
        "20. Hobbies/extra-curriculars\n"
        "21. Current location\n"
        
        "Return a JSON object with a mandatory 'classification' field and extract all fields as per instructions. "
        "Include profile_links, project_links, certificate_links, other_links from structured links. "
        "provide these links in the relevant sections (e.g., profile links in profile section, project links in projects to that particular project)."
        "Use empty string/list/null for missing data. Return valid JSON only.\n\n"
        f"Structured Links (pre-classified):\n{json.dumps(structured_links, indent=2)}\n\n"
        f"Resume Text:\n{resume_text}\n\n"
        "Use the structured links, experience, and internship data to fill in relevant fields. "
         "Use the experience and internship data to accurately populate job roles, periods, and durations."
        " Ensure no duplication of internships in experience."
        " Use the periods in experience and internships to calculate durations."
        f"Experience:\n{json.dumps(experiences, indent=2)}\n\n"
        f"Internships:\n{json.dumps(internships, indent=2)}\n\n"
    )

    url = f"{GROQ_API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"model": "llama-3.3-70b-versatile",
            "messages":[{"role":"system","content":prompt},{"role":"user","content":resume_text}],
            "temperature":0.0,"max_tokens":2000}

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            return json_match.group(0).strip()
        else:
            return json.dumps({"classification": "unknown", "error": "No valid JSON detected in API response"})
    except requests.exceptions.RequestException as e:
        return json.dumps({"classification": "unknown", "error": f"API request failed: {str(e)}"})

# ---------------- Self-Introduction Script ----------------
def generate_intro_script(parsed_resume):
    prompt = (
        "You are an AI assistant that creates a concise, professional self-introduction script for interviews. "
        "Use the following parsed resume data (JSON) to craft a natural, confident, and structured introduction. "
        "The script should last about 30–60 seconds (~150–250 words). "
        "Include all key categories: name, education, top technical skills, major projects or work experiences, and career goal. "
        "Summarize details intelligently: mention the most relevant information in each section, "
        "without listing every minor certification, hobby, or project. Focus on clarity and impact. "
        "Do NOT include raw JSON, headings, or explanations — return only the self-introduction as plain text."
        f"\n\nResume Data:\n{json.dumps(parsed_resume, indent=2)}"
    )
    url = f"{GROQ_API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"model":"llama-3.3-70b-versatile",
            "messages":[{"role":"system","content":"You are a career coach specializing in preparing candidates for interviews."},
                        {"role":"user","content":prompt}],
            "temperature":0.7,"max_tokens":300}
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        return f"Error generating script: {str(e)}"

# ---------------- Streamlit UI ----------------
st.title("Resume Parser + Self-Intro Generator with Image Detection")

uploaded_file = st.file_uploader("Upload your resume (PDF, DOCX, TXT)", type=["pdf","docx","txt"])

if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    file_path = os.path.join(UPLOAD_PATH, f"file.{file_ext}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Extract text
    data_obj = extract_text_and_links(file_path, file_type=file_ext)

    if data_obj["text"].startswith("__READ_ERROR__"):
        st.error(f"Failed to read file: {data_obj['text'][len('__READ_ERROR__:'):]}")
    else:
        # OCR check only for PDFs
        if file_ext == "pdf" and is_image_based_resume(file_path, data_obj["text"]):
            st.error("❌ Invalid resume detected. This resume appears to be fully or partially image-based (scanned). Please upload a text-based PDF, DOCX, or TXT file.")
        elif len(data_obj["text"].strip()) < 100:
            st.error("❌ Invalid resume detected. The file seems empty or unreadable. Please upload a text-based resume.")
        else:
            # Normal processing
            structured_links = classify_links(data_obj["links"])

            # Detect projects
            project_sections = []
            project_matches = re.findall(r"(Project\s*:\s*(.+?)\n(.*?)(?=\nProject|$))", data_obj["text"], re.S | re.I)
            for _, title, desc in project_matches:
                project_sections.append({"Title": title.strip(), "Description": desc.strip()})

            projects_with_links = assign_project_links(data_obj["text"], project_sections, data_obj["links"])
            structured_links["projects"] = projects_with_links

            # Detect experience and internships
            experience_matches = re.findall(r"(Experience|Work Experience)\s*:\s*(.+?)\n(?=(Experience|Projects|$))", data_obj["text"], re.S | re.I)
            experiences = [{"Description": text.strip(), "Period": ""} for _, text, _ in experience_matches]

            internship_matches = re.findall(r"(Internship)\s*:\s*(.+?)\n(?=(Internship|Projects|$))", data_obj["text"], re.S | re.I)
            internships = [{"Description": text.strip(), "Period": ""} for _, text, _ in internship_matches]

            ats_data = ats_extractor(data_obj["text"], structured_links, experiences, internships)

            try:
                ats_json = json.loads(ats_data)
                classification = ats_json.get("classification", "Unknown")
                st.subheader(f"Parsed Resume Data (Classification: {classification})")
                st.json(ats_json)

                if st.button("Generate Introduction Script"):
                    script = generate_intro_script(ats_json)
                    st.subheader("Generated Self-Introduction Script")
                    st.write(script)

                    # Generate audio
                    try:
                        audio_path = os.path.join(UPLOAD_PATH, "intro.wav")
                        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
                        tts.tts_to_file(text=script, file_path=audio_path)
                        st.audio(audio_path)
                    except Exception as e:
                        st.error(f"Text-to-Speech failed: {str(e)}")

            except json.JSONDecodeError:
                st.error("Failed to parse JSON data from extractor.")
                st.text(ats_data)
