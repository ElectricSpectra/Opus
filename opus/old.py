# --- main_app.py ---
import streamlit as st
import os
import tempfile
import subprocess
import gc
from typing import List, Optional, Dict, Any, Tuple

# --- Library Imports ---
# These are the core libraries for the application's functionality.
try:
    from faster_whisper import WhisperModel
    from faster_whisper.transcribe import Word
    import yt_dlp
except ImportError as e:
    st.error(f"A required library is missing: {e}")
    st.error("Please run: pip install faster-whisper yt-dlp")
    st.stop()

# --- FFMPEG Configuration & Verification ---
# We assume 'ffmpeg' is in the system's PATH, which is standard.
FFMPEG_EXECUTABLE = "ffmpeg"

@st.cache_data # Cache the result of this check
def verify_ffmpeg():
    """Checks if FFmpeg is installed and accessible."""
    try:
        # Run a silent version check to confirm ffmpeg's presence.
        result = subprocess.run(
            [FFMPEG_EXECUTABLE, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        st.error("🔴 FFMPEG is not found on your system.")
        st.error("Please install FFmpeg and ensure it's in your system's PATH.")
        st.code("On macOS (using Homebrew): brew install ffmpeg")
        st.code("On Windows (using Chocolatey): choco install ffmpeg")
        st.code("On Debian/Ubuntu: sudo apt update && sudo apt install ffmpeg")
        st.stop()
        return False

# --- AI Model and State Management ---
@st.cache_resource
def load_whisper_model() -> Optional[WhisperModel]:
    """
    Loads the faster-whisper model into memory.
    Using @st.cache_resource ensures this is only done once per session.
    """
    try:
        # Using a smaller, faster model for this CPU-based app.
        with st.spinner("Loading AI model for the first time... this may take a moment."):
            model = WhisperModel("base", device="cpu", compute_type="int8")
        return model
    except Exception as e:
        st.error(f"❌ Error loading faster-whisper model: {e}")
        return None

# --- Core Logic Functions ---

def download_youtube_video(url: str) -> Optional[Tuple[str, str]]:
    """
    Downloads a YouTube video to a temporary file and returns its path and title.
    Returns (None, None) on failure.
    """
    try:
        temp_dir = tempfile.gettempdir()
        # Create a unique but predictable filename template in the temp directory
        output_template = os.path.join(temp_dir, 'opus_clipper_%(id)s.%(ext)s')
        
        ydl_opts = {
            'format': 'best[height<=720][ext=mp4]/best[height<=720]/best',
            'outtmpl': output_template,
            'no_warnings': True,
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = info.get('requested_downloads')[0]['filepath']
            video_title = info.get('title', 'Unknown_Video_Title')
            
            if video_path and os.path.exists(video_path):
                return video_path, video_title
            else:
                return None, None
    except Exception as e:
        st.error(f"❌ Error during YouTube download: {e}")
        return None, None

def extract_audio(video_path: str) -> Optional[str]:
    """
    Extracts audio from a video file into a temporary WAV file for transcription.
    Returns the path to the audio file, or None on failure.
    """
    try:
        # Create a secure temporary file for the audio output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
            audio_path = temp_audio_file.name

        # FFmpeg command to extract and prepare audio for Whisper
        cmd = [
            FFMPEG_EXECUTABLE,
            '-i', video_path,
            '-vn',                # No video
            '-acodec', 'pcm_s16le', # Use a standard audio codec
            '-ar', '16000',       # Sample rate required by Whisper
            '-ac', '1',           # Mono audio
            '-y',                 # Overwrite output file if it exists
            audio_path
        ]
        
        # Use check=True to automatically raise an error if FFmpeg fails
        subprocess.run(cmd, check=True, capture_output=True)
        
        return audio_path
    except subprocess.CalledProcessError as e:
        st.error(f"❌ Audio extraction failed with FFmpeg: {e.stderr.decode()}")
        return None
    except Exception as e:
        st.error(f"❌ An error occurred during audio extraction: {e}")
        return None

def transcribe_audio(model: WhisperModel, audio_path: str) -> Optional[Dict[str, Any]]:
    """
    Transcribes audio using the loaded Whisper model.
    Returns a dictionary with the full text and a list of word objects with timestamps.
    """
    try:
        segments, _ = model.transcribe(audio_path, word_timestamps=True)
        
        all_words = []
        full_text = []
        for segment in segments:
            full_text.append(segment.text)
            for word in segment.words:
                all_words.append(word)

        if not all_words:
            return None

        return {'text': "".join(full_text).strip(), 'words': all_words}
    except Exception as e:
        st.error(f"❌ Transcription error: {e}")
        return None

def find_most_intense_segment(words: List[Word], clip_duration: int) -> Optional[Dict[str, Any]]:
    """

    Finds the segment with the most words within a given duration.
    This uses an efficient O(N) sliding window algorithm.
    """
    if not words:
        return None

    max_words_in_window = 0
    best_segment_info = None
    
    left = 0
    for right in range(len(words)):
        # Slide the left side of the window forward
        while words[right].start - words[left].start > clip_duration:
            left += 1
            
        # Check if the current window is the new best
        current_word_count = right - left + 1
        if current_word_count > max_words_in_window:
            max_words_in_window = current_word_count
            
            # Capture segment details
            segment_words = words[left : right + 1]
            best_segment_info = {
                "start": segment_words[0].start,
                "end": segment_words[-1].end,
                "text": " ".join(w.word for w in segment_words),
                "words": segment_words,
            }
            
    return best_segment_info

def escape_ffmpeg_path(path: str) -> str:
    """Prepares a file path for safe use in an FFmpeg filtergraph (crucial for Windows)."""
    if os.name == 'nt':
        return path.replace('\\', '\\\\').replace(':', '\\:')
    return path

def seconds_to_srt_time(seconds: float) -> str:
    """Converts seconds to the SRT timestamp format (HH:MM:SS,ms)."""
    # Ensure we don't have negative time
    seconds = max(0, seconds)
    
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def create_vertical_clip(video_path: str, start_time: float, end_time: float, words: List[Word]) -> Optional[str]:
    """
    Creates the final vertical video clip with burnt-in subtitles.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as output_file:
        output_path = output_file.name

    duration = end_time - start_time
    
    # Create SRT subtitle content
    subtitle_content = []
    subtitle_index = 1
    
    for word in words:
        word_start = max(0, word.start - start_time)
        word_end = min(duration, word.end - start_time)
        
        if word_start < duration and word_end > 0:
            clean_word = word.word.strip()
            if clean_word:
                # Format as SRT subtitle
                start_time_str = f"{int(word_start//3600):02d}:{int((word_start%3600)//60):02d}:{int(word_start%60):02d},{int((word_start%1)*1000):03d}"
                end_time_str = f"{int(word_end//3600):02d}:{int((word_end%3600)//60):02d}:{int(word_end%60):02d},{int((word_end%1)*1000):03d}"
                
                subtitle_content.append(f"{subtitle_index}")
                subtitle_content.append(f"{start_time_str} --> {end_time_str}")
                subtitle_content.append(clean_word)
                subtitle_content.append("")
                subtitle_index += 1
    
    # Write subtitle file
    with tempfile.NamedTemporaryFile(mode='w', suffix=".srt", delete=False, encoding='utf-8') as srt_file:
        srt_path = srt_file.name
        srt_file.write('\n'.join(subtitle_content))
    
    try:
        # CORRECT approach: Use subtitles filter to burn in SRT file
        # Escape the subtitle path for Windows
        escaped_srt_path = srt_path.replace('\\', '/').replace(':', '\\:')
        
        cmd = [
            FFMPEG_EXECUTABLE,
            '-i', video_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-vf', (
                f"scale=1080:-2,"
                f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,"
                f"subtitles='{escaped_srt_path}':force_style='Fontname=font,Fontsize=24,PrimaryColour=&H00ffffff,OutlineColour=&H00000000,Outline=2,Alignment=2'"
            ),
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '28',
            '-c:a', 'aac',
            '-y',
            output_path
        ]
        
        st.info(f"🔧 Creating video with {subtitle_index-1} subtitle entries")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Clean up subtitle file
        if os.path.exists(srt_path):
            os.remove(srt_path)
        
        return output_path
        
    except subprocess.CalledProcessError as e:
        st.error(f"❌ Video creation with subtitles failed. Trying fallback without subtitles...")
        st.error(f"FFmpeg error: {e.stderr}")
        
        # FALLBACK: Create video without any subtitles if subtitle approach fails
        try:
            fallback_cmd = [
                FFMPEG_EXECUTABLE,
                '-i', video_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-vf', 'scale=1080:-2,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '28',
                '-c:a', 'aac',
                '-y',
                output_path
            ]
            
            subprocess.run(fallback_cmd, check=True, capture_output=True, text=True)
            st.warning("⚠️ Created video without subtitles due to font/subtitle issues")
            
            # Clean up subtitle file
            if os.path.exists(srt_path):
                os.remove(srt_path)
            
            return output_path
            
        except subprocess.CalledProcessError as fallback_error:
            st.error(f"❌ Even fallback failed: {fallback_error.stderr}")
            if os.path.exists(srt_path):
                os.remove(srt_path)
            return None
            
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
        if os.path.exists(srt_path):
            os.remove(srt_path)
        return None


# --- Streamlit User Interface ---
st.set_page_config(layout="wide", page_title="AI Shorts Clipper")
st.title("🎬 AI-Powered YouTube Shorts Clipper")
st.markdown("This tool finds the most interesting segment of a video, reformats it for Shorts, and adds dynamic subtitles.")

# Run FFmpeg check at the start
verify_ffmpeg()

st.sidebar.header("How It Works")
st.sidebar.markdown("""
1.  **Paste a YouTube URL** of a talking-head video, podcast, or speech.
2.  **Adjust the desired clip duration.**
3.  **Click 'Generate Clip'.** The AI will find the most word-dense segment to use.
4.  **Review & Download** your short-form video, ready for upload!
""")

youtube_url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
clip_duration = st.slider("Clip Duration (seconds):", 15, 60, 30, 5)

# Set the ASS font directory to the current directory
os.environ["ASS_FONT_DIR"] = os.path.abspath(".")

if st.button("🚀 Generate Clip", type="primary", use_container_width=True):
    if youtube_url:
        # List to keep track of all temporary files for cleanup
        cleanup_files = []
        
        try:
            # --- Step 1: Download Video ---
            st.info("Step 1/5: Downloading YouTube video...")
            video_path, video_title = download_youtube_video(youtube_url)
            if not video_path: st.stop()
            cleanup_files.append(video_path)
            st.success(f"✅ Downloaded: '{video_title}'")

            # --- Step 2: Extract Audio ---
            st.info("Step 2/5: Extracting audio...")
            audio_path = extract_audio(video_path)
            if not audio_path: st.stop()
            cleanup_files.append(audio_path)
            st.success("✅ Audio extracted successfully.")

            # --- Step 3: Transcribe with AI ---
            st.info("Step 3/5: Transcribing with AI... (this may take a while)")
            model = load_whisper_model()
            if not model: st.stop()
            transcription = transcribe_audio(model, audio_path)
            if not transcription or not transcription['words']:
                st.error("❌ Could not transcribe audio or no speech was found.")
                st.stop()
            st.success("✅ Transcription complete!")
            with st.expander("View Full Transcription"):
                st.write(transcription['text'])

            # --- Step 4: Find Best Segment ---
            st.info("Step 4/5: Finding most intense segment...")
            best_segment = find_most_intense_segment(transcription['words'], clip_duration)
            if not best_segment:
                st.error("❌ Could not find a suitable speaking segment in the video.")
                st.stop()
            st.success(f"✅ Found best segment: {best_segment['start']:.1f}s - {best_segment['end']:.1f}s")

            # --- Step 5: Create Final Clip ---
            st.info("Step 5/5: Creating vertical clip with subtitles...")
            output_path = create_vertical_clip(
                video_path,
                best_segment['start'],
                best_segment['end'],
                best_segment['words']
            )
            if not output_path: st.stop()
            cleanup_files.append(output_path) # Add final video to cleanup list
            
            # --- Display Results ---
            st.header("✨ Your AI-Generated Short!", anchor=False)
            
            # Read the final video into memory so we can delete the file
            with open(output_path, "rb") as file:
                video_bytes = file.read()
            
            st.video(video_bytes)
            
            st.subheader("Clip Transcript:", anchor=False)
            st.markdown(f"> _{best_segment['text']}_")
            
            st.download_button(
                label="📥 Download Short",
                data=video_bytes,
                file_name=f"{video_title[:40]}_short.mp4",
                mime="video/mp4",
                use_container_width=True
            )

        except Exception as e:
            # Catch-all for any unexpected errors during the process
            st.error(f"An unexpected error occurred: {e}")
            
        finally:
            # --- Final Cleanup ---
            # Garbage collect to release model memory if possible
            gc.collect()
            for f_path in cleanup_files:
                if f_path and os.path.exists(f_path):
                    try:
                        os.remove(f_path)
                    except Exception:
                        # Silently ignore cleanup errors
                        pass
    else:
        st.warning("Please enter a YouTube URL to begin.")