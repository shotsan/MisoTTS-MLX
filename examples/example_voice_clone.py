import sys
import os
import time

# Ensure mlx_audio can be imported if this is run from the root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from mlx_generate_source import generate_audio
except ImportError:
    print("❌ ERROR: Could not import MisoTTS engine. Ensure you are running this from the root directory or your virtual environment is active.")
    sys.exit(1)

def main():
    print("=========================================")
    print("   MisoTTS Voice Cloning Example Script  ")
    print("=========================================\n")
    
    # 1. The text you want the AI to speak
    target_text = (
        "I think it's fundamentally important that we become a multi-planetary species. "
        "If you look at the trajectory of human history, we've always been expanding."
    )
    
    # 2. Path to the reference audio (10-15 seconds recommended)
    # We are using the bundled Elon Musk sample
    reference_audio = "./examples/elon_reference.wav"
    
    if not os.path.exists(reference_audio):
        print(f"❌ ERROR: Reference audio {reference_audio} not found!")
        sys.exit(1)
        
    print(f"[-] Target Text: {target_text}")
    print(f"[-] Reference Audio: {reference_audio}")
    print(f"[-] Expected Voice: Elon Musk\n")
    
    # 3. Setup core arguments
    tts_kwargs = {
        "text": target_text,
        "model": "mlx-community/MisoLabs-MisoTTS-bf16",
        
        # CRITICAL: If you want true voice cloning, do not pass "voice" 
        # or explicitly pass None so it overrides the "af_heart" fallback!
        "voice": None, 
        
        # Point to the audio you want to clone
        "ref_audio": reference_audio,
        
        # Auto-transcribe the reference audio by setting this to None
        # If Whisper fails, manually type the transcript of the reference audio here.
        "ref_text": None,
        
        # Formatting
        "output_path": "./",
        "file_prefix": f"elon_clone_example_{int(time.time())}",
        "audio_format": "wav",
        "join_audio": True,
    }
    
    print("[+] Initializing Engine... (This may take a minute)")
    
    try:
        generate_audio(**tts_kwargs)
        print(f"\n✅ Generation complete! Look for {tts_kwargs['file_prefix']}.wav in your root directory.")
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        print("Tip: Run `python debug_misotts.py` to diagnose MLX memory or audio length issues.")

if __name__ == "__main__":
    main()
