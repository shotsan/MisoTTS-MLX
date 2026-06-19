import sys
import subprocess
import os

# --- Helper to check system output ---
def run_command(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return None

def debug_gpu_memory():
    print("--- [1] Checking macOS GPU Wired Memory Limit ---")
    output = run_command(["sysctl", "iogpu.wired_limit_mb"])
    if output:
        try:
            limit = int(output.split(":")[-1].strip())
            if limit < 22000:
                print(f"❌ WARNING: Your GPU wired memory limit is dangerously low ({limit} MB).")
                print("   The 16-bit 8B model requires at least ~22GB of un-wired GPU memory.")
                print("   Fix it by running: sudo sysctl iogpu.wired_limit_mb=22528")
            else:
                print(f"✅ GPU Memory limit is healthy ({limit} MB).")
        except ValueError:
            print("⚠️ Could not parse iogpu.wired_limit_mb output.")
    else:
        print("⚠️ Failed to check iogpu.wired_limit_mb. Make sure you are on Apple Silicon.")

def debug_audio_file(audio_path):
    print(f"\n--- [2] Analyzing Reference Audio: {audio_path} ---")
    if not os.path.exists(audio_path):
        print(f"❌ ERROR: Audio file '{audio_path}' does not exist.")
        return

    # Use ffprobe to get duration
    output = run_command([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path
    ])
    
    if output:
        try:
            duration = float(output)
            print(f"[-] Audio Duration: {duration:.2f} seconds")
            if duration > 16.0:
                print(f"❌ WARNING: Audio is too long! Anything over 15 seconds can deadlock the MLX KV Cache.")
                print(f"   Please truncate it using: ffmpeg -y -i {audio_path} -t 15 -c copy short_audio.wav")
            elif duration < 5.0:
                print(f"⚠️ WARNING: Audio is very short! It may not provide enough vocal data for a good clone.")
                print(f"   Consider looping it with ffmpeg.")
            else:
                print(f"✅ Audio duration is in the optimal sweet spot (5-15 seconds).")
        except ValueError:
            print("⚠️ Could not parse audio duration.")
    else:
        print("⚠️ ffprobe not installed or failed to read file. (Install via 'brew install ffmpeg')")

def debug_whisper_transcription(audio_path):
    print(f"\n--- [3] Testing Whisper Auto-Transcription ---")
    print(f"[-] Loading Whisper model to test how the engine will perceive your audio...")
    try:
        from mlx_audio.tts.utils import load_model
        import mlx.core as mx
        from mlx_audio.utils import load_audio
    except ImportError:
        print("❌ ERROR: mlx_audio not found. Run this in your .venv")
        return

    try:
        stt_model = load_model("mlx-community/whisper-large-v3-turbo-asr-fp16")
        audio_data = load_audio(audio_path)
        result = stt_model.generate(audio_data)
        transcript = result.text.strip()
        print(f"\n[Whisper Output]: \"{transcript}\"")
        print(f"\n✅ If this transcript perfectly matches what is actually said in {audio_path}, you are good to go!")
        print(f"❌ If Whisper dropped sentences (e.g. because of echoes or noise), the voice clone WILL FAIL and fallback to a default voice.")
        print(f"   If so, you must manually inject the correct text into the 'ref_text' parameter in your script.")
    except Exception as e:
        print(f"❌ Failed to run Whisper test: {e}")

def debug_target_text(target_text):
    print(f"\n--- [4] Analyzing Target Output Text ---")
    word_count = len(target_text.split())
    print(f"[-] Word count: {word_count}")
    if word_count > 60:
        print(f"❌ WARNING: Your target text is very long ({word_count} words).")
        print(f"   The 16-bit model may silent-deadlock the KV cache while generating this much audio.")
        print(f"   Consider splitting it into chunks of ~50 words.")
    else:
        print(f"✅ Target text length is safe.")

if __name__ == "__main__":
    print("=========================================")
    print("   MisoTTS MLX Diagnostics & Debugger    ")
    print("=========================================\n")
    
    debug_gpu_memory()
    
    # You can change these to test specific files
    test_audio = "elon_short.wav"
    test_text = "I think it's fundamentally important that we become a multi-planetary species."
    
    debug_audio_file(test_audio)
    debug_target_text(test_text)
    debug_whisper_transcription(test_audio)
    
    print("\n[+] Diagnostics complete.")
