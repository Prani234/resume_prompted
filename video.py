import os
import streamlit as st
import PyPDF2
import json
import requests
import re
import textwrap
from dotenv import load_dotenv
from TTS.api import TTS
from pydub import AudioSegment
from PIL import Image, ImageDraw, ImageFont
import subprocess

# Load environment variables
load_dotenv()

# API Configuration
api_key = os.environ.get("GROQ_API_KEY")
GROQ_API_BASE = "https://api.groq.com/openai/v1"

# Directory for uploads
UPLOAD_PATH = "__DATA__"
os.makedirs(UPLOAD_PATH, exist_ok=True)

# ---------------- PDF Handling ----------------
def extract_text_and_links(path):
    try:
        text = ""
        links = []

        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

                if "/Annots" in page:
                    for annot in page["/Annots"]:
                        obj = annot.get_object()
                        if "/A" in obj and "/URI" in obj["/A"]:
                            links.append(obj["/A"]["/URI"])

        return {"text": text.strip(), "links": sorted(set(links))}

    except Exception as e:
        return {"text": f"__PDF_READ_ERROR__:{str(e)}", "links": []}

# ---------------- Link Classification ----------------
def classify_links(links):
    profile_patterns = [r'linkedin\.com', r'github\.com$', r'behance\.net', r'personalwebsite\.com']
    certificate_patterns = [r'certificates\.edu', r'docsend\.com', r'publication\.org']

    classified = {"profile_links": [], "certificate_links": [], "other_links": []}

    for link in links:
        if any(re.search(p, link, re.I) for p in profile_patterns):
            classified["profile_links"].append(link)
        elif any(re.search(p, link, re.I) for p in certificate_patterns):
            classified["certificate_links"].append(link)
        else:
            classified["other_links"].append(link)

    for k in classified:
        classified[k] = sorted(set(classified[k]))
    return classified

# ---------------- Project-specific Link Assignment ----------------
def assign_project_links(resume_text, project_sections, all_links):
    projects_with_links = []
    for project in project_sections:
        description = project.get("Description", "")
        project_links = [link for link in all_links if link in description]
        project["project_links"] = project_links
        projects_with_links.append(project)
    return projects_with_links

# ---------------- ATS Extractor ----------------
def ats_extractor(resume_text, structured_links):
    prompt = (
       "You are an AI bot parsing resumes with high accuracy (target 95%). "
       "First, determine if the resume belongs to a **Fresher** (no professional work experience beyond internships, "
        "recent graduation within 2 years, focus on education/projects) or an **Experienced professional** "
        "(mentions job roles, companies, and total experience > 1 year). Use these criteria strictly:\n"
        "- Fresher: Internship duration < 6 months, no full-time roles, recent education (e.g., 2023-2025).\n"
        "- Experienced: Full-time roles with total experience > 1 year, or senior roles.\n"
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

         "If Fresher, extract:\n"
        "1. Name\n2. Email\n3. Phone number\n4. Profile links (LinkedIn, GitHub, etc)\n"
        "5. Education details (list of {Degree, College, Year, Grades})\n6. Technical skills (include skills mentioned in Skills section, job descriptions, project descriptions,Certifications and summary)\n"
        "7. Soft skills (list)\n8. Projects (list of {Title, Description, Role,Technologies used, Responsibilities,project links if provided})\n"
        "9. Certifications (list)\n10. Languages known (list)\n11. Hobbies/extra-curriculars\n"
        "12. Current location\n13. Expected CTC\n14. Work/location preference\n15. work mode\n"
        "16. Internship details (list of {Company, Duration, Role,Technologies used in each role, Responsibilities})\n17. Summary\n\n"

        "If Experienced, extract:\n"
        "1. Name\n2. Email\n3. Phone number\n4. Profile links (LinkedIn, GitHub, etc)\n"
        "5. Technical skills (include skills mentioned in Skills section, job descriptions, project descriptions,Certifications and summary)\n6. Soft skills (list)\n"
        "7. Total Experience\n"  
        "8. Experience details (list of {Company, Role, Duration, Technologies used, Responsibilities})\n"
        "    - Only full-time professional roles. Do not include internships here.\n"
        "9. Internship details (list of {Company, Duration, Role,Technologies used, Responsibilities})\n"
        "    - Only internships. Do not duplicate in Experience details.\n"
        "10. Projects (list of {Title, Description, Role,Technologies used, Responsibilities,project links if provided})\n"
        "11. Certifications (list)\n"
        "12. Education details (list of {Degree, College, Year, Grades})\n"
        "13. Current CTC\n14. Expected CTC\n"
        "15. Notice period\n16. Work mode\n17. Preferred location\n"
        "18. Summary\n19. Languages known (list)\n"
        
        "Return a JSON object with a mandatory 'classification' field and extract all fields as per instructions. "
        "Include profile_links, project_links, certificate_links, other_links from structured links. "
        "provide these links in the relevant sections (e.g., profile links in profile section, project links in projects to that particular project)."
        "Use empty string/list/null for missing data. Return valid JSON only.\n\n"
        f"Structured Links (pre-classified):\n{json.dumps(structured_links, indent=2)}\n\n"
        f"Resume Text:\n{resume_text}\n\n"
    )

    url = f"{GROQ_API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": resume_text}],
        "temperature": 0.0,
        "max_tokens": 2000,
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            return json_match.group(0).strip()
        else:
            return json.dumps({"classification": "unknown", "error": "No valid JSON detected"})
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
    data = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a career coach."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 300,
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        return f"Error generating script: {str(e)}"

# ---------------- Typing Animation (Synced to Audio) ----------------
def animate_text(script, audio_path, frame_size=(720, 480), line_height=40, fps=30, out_dir="__DATA__/frames"):
    os.makedirs(out_dir, exist_ok=True)

    # Wrap text
    wrapped_lines = []
    for line in script.split("\n"):
        wrapped_lines.extend(textwrap.wrap(line, width=60))

    # Audio
    audio = AudioSegment.from_file(audio_path)
    audio_duration = audio.duration_seconds  # seconds
    total_chars = sum(len(line) for line in wrapped_lines)
    char_duration = audio_duration / max(1, total_chars)  # seconds per character
    frames_per_char = max(1, int(fps * char_duration))

    fnt = ImageFont.load_default()
    frame = 0

    for line_idx, input_string in enumerate(wrapped_lines):
        y_position = frame_size[1]//2 - line_height*(len(wrapped_lines)//2) + line_idx*line_height
        for i in range(len(input_string)+1):
            for _ in range(frames_per_char):
                img = Image.new("RGB", frame_size, color="black")
                d = ImageDraw.Draw(img)

                # Draw all previous lines fully
                for j in range(line_idx):
                    y_prev = frame_size[1]//2 - line_height*(len(wrapped_lines)//2) + j*line_height
                    d.text((50, y_prev), wrapped_lines[j], font=fnt, fill=(255,255,255))

                # Draw current line partially
                d.text((50, y_position), input_string[:i], font=fnt, fill=(255,255,255))
                img.save(f"{out_dir}/frame_{frame:05}.png")
                frame += 1

    return out_dir, frame_size, fps

# ---------------- Create Video ----------------
def create_video_from_frames_ffmpeg(frames_dir, fps, output_path, audio_path):
    # Build ffmpeg command
    command = [
        "ffmpeg", "-y",
        "-r", str(fps),
        "-i", os.path.join(frames_dir, "frame_%05d.png"),
        "-i", audio_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        output_path
    ]
    subprocess.run(command, check=True)

# ---------------- Streamlit UI ----------------
st.title("Resume Parser + Self-Intro Video Generator")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    file_path = os.path.join(UPLOAD_PATH, "file.pdf")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    data_obj = extract_text_and_links(file_path)

    if data_obj["text"].startswith("__PDF_READ_ERROR__:"):
        st.error(f"Failed to read PDF: {data_obj['text'][len('__PDF_READ_ERROR__:'):]}")
    else:
        structured_links = classify_links(data_obj["links"])

        # Extract project sections
        project_sections = []
        project_matches = re.findall(r"(Project\s*:\s*(.+?)\n(.*?)(?=\nProject|$))", data_obj["text"], re.S | re.I)
        for _, title, desc in project_matches:
            project_sections.append({"Title": title.strip(), "Description": desc.strip()})

        projects_with_links = assign_project_links(data_obj["text"], project_sections, data_obj["links"])
        structured_links["projects"] = projects_with_links

        ats_data = ats_extractor(data_obj["text"], structured_links)

        try:
            ats_json = json.loads(ats_data)
            classification = ats_json.get("classification", "Unknown")
            st.subheader(f"Parsed Resume Data (Classification: {classification})")
            st.json(ats_json)

            if st.button("Generate Introduction Video"):
                script = generate_intro_script(ats_json)
                st.subheader("Generated Self-Introduction Script")
                st.write(script)

                try:
                    audio_path = os.path.join(UPLOAD_PATH, "intro.wav")
                    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
                    tts.tts_to_file(text=script, file_path=audio_path)

                    frames_dir, frame_size, fps = animate_text(script, audio_path)
                    final_video = os.path.join(UPLOAD_PATH, "intro_final.mp4")

                    create_video_from_frames_ffmpeg(frames_dir, fps, final_video, audio_path)

                    st.subheader("Generated Self-Introduction Video (Synced with Audio)")
                    st.video(final_video)

                except Exception as e:
                    st.error(f"Video generation failed: {str(e)}")

        except json.JSONDecodeError:
            st.error("Failed to parse JSON data from extractor.")
            st.text(ats_data)
