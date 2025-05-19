import os
import tempfile
import gradio as gr
import torch
import numpy as np
import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
import warnings
import gc
from dotenv import load_dotenv
import subprocess
import sys
import datetime

# Load environment variables from .env file
load_dotenv()

# For YouTube support - only using yt-dlp
try:
    import yt_dlp
except ImportError:
    print("yt-dlp not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
    import yt_dlp

# Suppress warnings
warnings.filterwarnings("ignore")

# Check if CUDA is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "float32"

# Get HF token from environment variable if available
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")

# Default output directory for transcriptions
DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Documents", "WhisperX_Transcriptions")
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

def format_result(result):
    """Format the diarized transcript for display"""
    formatted_text = ""
    
    for segment in result["segments"]:
        speaker = segment.get("speaker", "Unknown")
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        
        timestamp = f"[{format_timestamp(start)} → {format_timestamp(end)}]"
        formatted_text += f"**Speaker {speaker}** {timestamp}\n{text}\n\n"
    
    return formatted_text

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def format_srt(result):
    """Format the diarized transcript in SRT format"""
    srt_content = ""
    segment_count = 1
    
    for segment in result["segments"]:
        speaker = segment.get("speaker", "Unknown")
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        
        # Format timestamps for SRT (HH:MM:SS,mmm)
        start_time = format_srt_timestamp(start)
        end_time = format_srt_timestamp(end)
        
        # Add segment to SRT content
        srt_content += f"{segment_count}\n"
        srt_content += f"{start_time} --> {end_time}\n"
        srt_content += f"[Speaker {speaker}] {text}\n\n"
        
        segment_count += 1
    
    return srt_content

def format_srt_timestamp(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_whole = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_whole:02d},{milliseconds:03d}"

def save_transcript_to_file(transcript, output_dir, filename=None, format_type="txt"):
    """Save transcript to a file in specified format"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    if not filename:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcript_{timestamp}"
    
    # Remove any existing extension from filename
    filename = os.path.splitext(filename)[0]
    
    # Add appropriate extension based on format type
    if format_type.lower() == "srt":
        filename += ".srt"
    else:
        filename += ".txt"
    
    output_path = os.path.join(output_dir, filename)
    
    # Save the transcript
    with open(output_path, 'w', encoding='utf-8') as f:
        if format_type.lower() == "txt":
            # Remove markdown formatting for file output
            clean_transcript = transcript.replace('**', '')
            f.write(clean_transcript)
        else:
            # SRT format is already properly formatted
            f.write(transcript)
    
    return output_path

def save_hf_token(token):
    """Save HuggingFace token to .env file"""
    if not token:
        return
    
    # Read existing .env file
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    env_content = {}
    
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    env_content[key] = value
    
    # Update token
    env_content['HUGGINGFACE_TOKEN'] = token
    
    # Write back to .env file
    with open(env_path, 'w') as f:
        for key, value in env_content.items():
            f.write(f"{key}={value}\n")
    
    return token

def verify_hf_token(token):
    """Verify if the HuggingFace token is valid"""
    import requests
    
    try:
        response = requests.get(
            "https://huggingface.co/api/models/pyannote/speaker-diarization",
            headers={"Authorization": f"Bearer {token}"}
        )
        return response.status_code == 200
    except Exception:
        return False

def download_youtube_audio(url, output_dir=None):
    """Download audio from YouTube URL using yt-dlp"""
    print(f"Downloading YouTube audio from: {url}")
    
    # Use the specified output directory or temp directory
    if output_dir and os.path.isdir(output_dir):
        temp_dir = output_dir
    else:
        temp_dir = tempfile.gettempdir()
    
    try:
        print("Downloading with yt-dlp...")
        # Extract video ID from URL
        if "youtu.be" in url:
            video_id = url.split("/")[-1].split("?")[0]
        elif "youtube.com" in url:
            if "v=" in url:
                video_id = url.split("v=")[1].split("&")[0]
            else:
                # For other YouTube URL formats
                video_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            # For non-YouTube URLs
            video_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a unique filename with full path
        output_filename = f"ytdlp_{video_id}"
        output_path = os.path.join(temp_dir, output_filename)
        
        # Configure yt-dlp options - key change is in the outtmpl format
        ydl_opts = {
            'format': 'bestaudio/best',
            'paths': {'home': temp_dir},
            'outtmpl': {'default': output_path},
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'verbose': True
        }
        
        # Download the audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', video_id)
        
        # The actual output file will have .mp3 extension after FFmpeg processing
        expected_file = f"{output_path}.mp3"
        
        # Check if file exists
        if not os.path.exists(expected_file):
            # Try to find any file that starts with our output_path
            possible_files = [f for f in os.listdir(temp_dir) if f.startswith(os.path.basename(output_path))]
            if possible_files:
                # Use the first matching file
                expected_file = os.path.join(temp_dir, possible_files[0])
                print(f"Found alternative file: {expected_file}")
            else:
                raise FileNotFoundError(f"Downloaded file not found at {expected_file} or with similar name")
        
        print(f"Downloaded YouTube audio: {title} to {expected_file}")
        return expected_file
    
    except Exception as e:
        print(f"yt-dlp download failed: {str(e)}")
        # Try a simpler approach as fallback
        try:
            print("Trying fallback download method...")
            fallback_output = os.path.join(temp_dir, f"fallback_{video_id}.mp3")
            
            # Simpler options without post-processing
            simple_opts = {
                'format': 'bestaudio/best',
                'outtmpl': fallback_output,
                'verbose': True,
                'no_warnings': False
            }
            
            with yt_dlp.YoutubeDL(simple_opts) as ydl:
                ydl.download([url])
            
            if os.path.exists(fallback_output):
                print(f"Fallback download successful: {fallback_output}")
                return fallback_output
        except Exception as fallback_error:
            print(f"Fallback download also failed: {str(fallback_error)}")
        
        raise Exception(f"Failed to download YouTube audio: {str(e)}")

def transcribe_with_diarization(
    audio_file, 
    language=None, 
    model_size="medium", 
    hf_token=None,
    min_speakers=None,
    max_speakers=None,
    batch_size=16,
    output_dir=None,
    custom_align_model=None
):
    """
    Transcribe audio with speaker diarization
    
    Args:
        audio_file: Path to audio file or YouTube URL
        language: Language code (optional, will auto-detect if not provided)
        model_size: WhisperX model size
        hf_token: HuggingFace token for diarization model
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers
        batch_size: Batch size for transcription
        output_dir: Directory to save the transcript
        
    Returns:
        Formatted transcript with speaker labels and file path
    """
    # Use token from environment variable if not provided
    if not hf_token and HF_TOKEN:
        hf_token = HF_TOKEN
    elif hf_token:
        # Save token for future use
        save_hf_token(hf_token)
    
    # Use default output directory if not specified
    if not output_dir:
        output_dir = DEFAULT_OUTPUT_DIR
    
    # Convert language selection to proper language code
    if language == "Auto-detect":
        language = None
    elif language in ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Dutch", "Japanese", "Chinese", "Russian", "Korean", "Hindi"]:
        # Convert language name to ISO code
        language_map = {
            "English": "en", "Spanish": "es", "French": "fr", "German": "de",
            "Italian": "it", "Portuguese": "pt", "Dutch": "nl", "Japanese": "ja",
            "Chinese": "zh", "Russian": "ru", "Korean": "ko", "Hindi": "hi"
        }
        language = language_map.get(language, None)
    
    try:
        # Check if input is a YouTube URL
        if isinstance(audio_file, str) and ('youtube.com' in audio_file or 'youtu.be' in audio_file):
            # Use yt-dlp to download YouTube audio
            audio_file = download_youtube_audio(audio_file, output_dir)
        
        # 1. Load audio
        print("Loading audio...")
        audio = whisperx.load_audio(audio_file)
        
        # 2. Transcribe with WhisperX
        print("Transcribing with WhisperX...")
        model = whisperx.load_model(
            model_size, 
            device=DEVICE, 
            compute_type=COMPUTE_TYPE,
            language=language
        )
        
        result = model.transcribe(
            audio, 
            batch_size=batch_size,
            print_progress=True
        )
        
        # Free up memory
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        # 3. Align whisper output
        print("Aligning transcript...")
        try:
            # Use custom alignment model if provided
            if custom_align_model and isinstance(custom_align_model, str) and custom_align_model.strip():
                print(f"Using custom alignment model: {custom_align_model}")
                model_a, metadata = whisperx.load_align_model(
                    language_code=result["language"],
                    device=DEVICE,
                    model_name=custom_align_model.strip()
                )
            else:
                model_a, metadata = whisperx.load_align_model(
                    language_code=result["language"],
                    device=DEVICE
                )
            
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device=DEVICE,
                print_progress=True
            )
            
            # Free up memory
            del model_a
            gc.collect()
            torch.cuda.empty_cache()
        except ValueError as e:
            # Handle the case where there's no alignment model for the language
            print(f"Alignment model error: {str(e)}")
            print("Continuing without alignment...")
            # We can still continue with diarization using the unaligned segments
            # No need to modify the result object as it already contains the segments
        
        # 4. Speaker diarization
        print("Performing speaker diarization...")
        try:
            # Initialize diarization pipeline without the 'model' parameter
            diarize_model = DiarizationPipeline(
                use_auth_token=hf_token,
                device=DEVICE
            )
            
            diarize_segments = diarize_model(
                audio_file,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            # 5. Assign speakers to words
            result = assign_word_speakers(diarize_segments, result)
            
            # 6. Format output
            output_text = format_result(result)
            
            # Format SRT output
            srt_text = format_srt(result)
            
        except Exception as e:
            print(f"Diarization error: {str(e)}")
            print("Continuing without speaker diarization...")
            # Return the transcription without diarization
            output_text = "**Speaker Diarization Failed**\n\n"
            output_text += "Transcription without speaker identification:\n\n"
            # Create SRT without speaker diarization
            srt_text = ""
            segment_count = 1
            for segment in result["segments"]:
                start = segment["start"]
                end = segment["end"]
                text = segment["text"]
                start_time = format_srt_timestamp(start)
                end_time = format_srt_timestamp(end)
                srt_text += f"{segment_count}\n"
                srt_text += f"{start_time} --> {end_time}\n"
                srt_text += f"{text}\n\n"
                segment_count += 1
        
        # 7. Save transcript to files
        base_filename = os.path.basename(audio_file).split('.')[0]
        
        # Save TXT format
        txt_filename = f"{base_filename}_transcript.txt"
        txt_file_path = save_transcript_to_file(output_text, output_dir, txt_filename, "txt")
        
        # Save SRT format
        srt_filename = f"{base_filename}_transcript.srt"
        srt_file_path = save_transcript_to_file(srt_text, output_dir, srt_filename, "srt")
        
        # Add file path information to the output
        output_text += f"\n\n---\n**Transcript saved to:**\n"
        output_text += f"- Text format: {txt_file_path}\n"
        output_text += f"- SRT format: {srt_file_path}"
        
        return output_text
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details: {error_details}")
        return f"Error: {str(e)}\n\nPlease check if you've provided a valid input and HuggingFace token."

def create_interface():
    # Add function to clear GPU memory
    def clear_gpu_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except:
                pass
        return "Memory cleared"
    
    with gr.Blocks(title="WhisperX Transcription with Speaker Diarization") as app:
        gr.Markdown("# 🎙️ WhisperX Transcription with Speaker Diarization")
        gr.Markdown("Upload an audio file or provide a YouTube URL to transcribe with speaker diarization")
        
        with gr.Row():
            with gr.Column():
                # Create tabs for different input methods
                with gr.Tabs():
                    with gr.TabItem("Upload Audio"):
                        audio_input = gr.Audio(type="filepath", label="Upload Audio File")
                    
                    with gr.TabItem("YouTube URL"):
                        youtube_url = gr.Textbox(
                            label="YouTube URL",
                            placeholder="Enter YouTube URL (e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ)"
                        )
                
                with gr.Row():
                    language = gr.Dropdown(
                        choices=["Auto-detect", "English", "Spanish", "French", "German", 
                                "Italian", "Portuguese", "Dutch", "Japanese", "Chinese", 
                                "Russian", "Korean", "Hindi", "Malay"],
                        value="Auto-detect",
                        label="Language"
                    )
                    
                    model_size = gr.Dropdown(
                        choices=["tiny", "base", "small", "medium", "large"],
                        value="medium",
                        label="Model Size"
                    )
                
                # Add custom alignment model option
                custom_align_model = gr.Textbox(
                    label="Custom Alignment Model (optional)",
                    placeholder="e.g., malaysian/wav2vec2-large-xlsr-53-malay",
                    value=""
                )
                
                with gr.Row():
                    min_speakers = gr.Number(label="Min Speakers (Optional)", precision=0)
                    max_speakers = gr.Number(label="Max Speakers (Optional)", precision=0)
                    batch_size = gr.Slider(minimum=1, maximum=32, value=16, step=1, label="Batch Size")
                
                # Add output directory selection
                output_dir = gr.Textbox(
                    label="Output Directory (where transcripts will be saved)",
                    placeholder="Leave empty for default location",
                    value=DEFAULT_OUTPUT_DIR
                )
                
                hf_token = gr.Textbox(
                    label="HuggingFace Token (for diarization model)",
                    placeholder="Enter your HuggingFace token here",
                    type="password",
                    value=HF_TOKEN if HF_TOKEN else None
                )
                
                submit_btn = gr.Button("Transcribe", variant="primary")
            
            with gr.Column():
                output = gr.Markdown(label="Transcription Output")
                progress = gr.Textbox(label="Progress", value="Ready")
        
        def process_with_progress(audio_input, youtube_url, language, model_size, hf_token, min_speakers, max_speakers, batch_size, output_dir, custom_align_model):
            progress_updates = []
            
            def update_progress(msg):
                progress_updates.append(msg)
                return "\n".join(progress_updates)
            
            update_progress("Starting transcription process...")
            
            # Verify output directory
            if output_dir and not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    yield update_progress(f"Created output directory: {output_dir}"), ""
                except Exception as e:
                    yield update_progress(f"Error creating output directory: {str(e)}. Using default directory."), ""
                    output_dir = DEFAULT_OUTPUT_DIR
            
            # Verify HuggingFace token
            if hf_token:
                yield update_progress("Verifying HuggingFace token..."), ""
                if not verify_hf_token(hf_token):
                    yield update_progress("Warning: HuggingFace token verification failed. Speaker diarization may not work."), ""
            
            yield update_progress("Preparing input..."), ""
            
            # Determine which input to use
            input_source = youtube_url if youtube_url and youtube_url.strip() else audio_input
            if not input_source:
                yield update_progress("Error: No input provided. Please upload an audio file or enter a YouTube URL."), ""
                return
            
            # Start transcription
            yield update_progress(f"Starting transcription with {model_size} model..."), ""
            
            try:
                result = transcribe_with_diarization(
                    input_source,
                    language=language,
                    model_size=model_size,
                    hf_token=hf_token,
                    min_speakers=min_speakers if min_speakers and min_speakers > 0 else None,
                    max_speakers=max_speakers if max_speakers and max_speakers > 0 else None,
                    batch_size=batch_size,
                    output_dir=output_dir,
                    custom_align_model=custom_align_model.value if hasattr(custom_align_model, 'value') else custom_align_model
                )
                
                yield update_progress("Transcription complete!"), result
                
            except Exception as e:
                error_message = str(e)
                yield update_progress(f"Error during transcription: {error_message}"), f"**Error:** {error_message}"
        
        # Connect the interface components
        submit_btn.click(
            fn=process_with_progress,
            inputs=[audio_input, youtube_url, language, model_size, hf_token, min_speakers, max_speakers, batch_size, output_dir, custom_align_model],
            outputs=[progress, output]
        )
        
        # Add memory clearing button
        clear_memory_btn = gr.Button("Clear GPU Memory")
        clear_memory_btn.click(fn=clear_gpu_memory, inputs=[], outputs=gr.Textbox(label="Memory Status"))
        
        return app

# Create and launch the interface
if __name__ == "__main__":
    app = create_interface()
    app.launch(share=False)
