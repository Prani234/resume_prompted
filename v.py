import streamlit as st
import tempfile
import os
from manim import *
from pydub import AudioSegment
import re
import numpy as np

st.title("Instagram-Style Continuous Scrolling Video Generator")
st.write("Paste your text and upload audio. Sentences scroll continuously in sync with the audio, like Instagram Reels.")

# Inputs
text_input = st.text_area("Paste your text here:")
audio_file = st.file_uploader("Upload Audio (mp3 or wav)", type=["mp3", "wav"])

# Optional: Input for sentence timings
st.write("Optionally, provide timings for each sentence (in seconds, comma-separated). Leave blank for automatic estimation.")
timings_input = st.text_input("Sentence timings (e.g., 2.5, 4.0, 6.5):")

if st.button("Generate Video") and text_input.strip() and audio_file:
    # Save audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_audio:
        tmp_audio.write(audio_file.read())
        audio_path = tmp_audio.name

    # Convert audio to wav if needed
    audio_seg = AudioSegment.from_file(audio_path)
    if audio_file.type == "audio/mpeg":
        audio_path_wav = audio_path.replace(".mp3", ".wav")
        audio_seg.export(audio_path_wav, format="wav")
    else:
        audio_path_wav = audio_path

    total_duration = audio_seg.duration_seconds
    sentences = re.split(r'(?<=[.!?]) +', text_input.strip())

    # Estimate or use provided timings
    if timings_input.strip():
        sentence_timings = [float(t.strip()) for t in timings_input.split(",")]
        if len(sentence_timings) != len(sentences):
            st.error("Number of timings must match number of sentences.")
            st.stop()
        sentence_timings.append(total_duration)
    else:
        sentence_timings = np.linspace(0, total_duration, len(sentences) + 1).tolist()

    output_dir = tempfile.mkdtemp()
    manim_file_path = os.path.join(output_dir, "smooth_scroll.py")

    with open(manim_file_path, "w") as f:
        f.write(f"""
from manim import *

class SmoothScrolling(Scene):
    def construct(self):
        sentences = {sentences}
        total_duration = {total_duration}

        self.camera.background_color = BLACK

        # Create Text objects stacked vertically
        texts = VGroup(*[Text(s, font='Arial', font_size=48, weight=BOLD, color=WHITE).scale(0.8) for s in sentences])
        texts.arrange(DOWN, buff=0.5)
        texts.move_to(DOWN * 3)  # Start below the screen
        self.add(texts)

        # Total scroll distance
        scroll_distance = 6 + texts.height  # scroll from bottom to top
        self.play(
            texts.animate.shift(UP * scroll_distance),
            run_time=total_duration,
            rate_func=linear
        )

# Configure audio
config.media_dir = "{output_dir}"
config.audio = "{audio_path_wav}"
""")

    st.write("Rendering continuous scrolling video with Manim...")
    os.system(f"manim -ql {manim_file_path} SmoothScrolling -o {output_dir}/scroll_video")

    video_path = os.path.join(output_dir, "scroll_video.mp4")
    final_output = os.path.join(output_dir, "final_video.mp4")

    if os.path.exists(video_path):
        os.system(f'ffmpeg -y -i "{video_path}" -i "{audio_path_wav}" -c:v copy -c:a aac "{final_output}"')

    if os.path.exists(final_output):
        st.video(final_output)
        with open(final_output, "rb") as f:
            st.download_button(
                "⬇️ Download Video",
                f,
                file_name="instagram_reels_scroll.mp4",
                mime="video/mp4"
            )
    else:
        st.error("Video generation failed. Check Manim and ffmpeg logs.")

    os.remove(audio_path)
    if audio_path_wav != audio_path:
        os.remove(audio_path_wav)

else:
    if not text_input.strip():
        st.warning("Please provide text input.")
    if not audio_file:
        st.warning("Please upload an audio file.")
