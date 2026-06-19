import sys
import os
import time
from mlx_audio.tts.generate import generate_audio
from mlx_audio.utils import load_model

import mlx.core as mx

# Lock MLX into the newly unlocked unswappable GPU memory!
mem_limit = 22 * 1024 * 1024 * 1024
mx.set_wired_limit(mem_limit)
mx.set_cache_limit(mem_limit)

def main():
    # 1. Define all your parameters here for easy tweaking.
    # We've listed the most important ones for you with their defaults.
    
    # The text you want the AI to speak
    text = (
        "I think it's fundamentally important that we become a multi-planetary species. If you look at the trajectory of human history, we've always been expanding. But eventually, if we don't extend life beyond Earth, some extinction event will wipe us out. That's why SpaceX is so focused on making life multi-planetary, specifically Mars."
    )
    
    # Model selection
    model_id = "mlx-community/MisoLabs-MisoTTS-bf16"
    
    print(f"[+] Setting up TTS generator...")
    print(f"[-] Model: {model_id}")
    print(f"[-] Text: {text}")
    print(f"[-] Verbose mode is OFF. The messy logs are hidden.")
    
    # === Core TTS Arguments ===
    tts_kwargs = {
        "text": text,
        "model": model_id,               # Path or HuggingFace repo ID
        "speed": 1.0,                    # Audio playback speed (1.0 is normal)
        "lang_code": "en",               # Language code ("en" for English)
        "temperature": 0.7,              # Generation temperature (controls randomness)
        "max_tokens": 1200,              # Max length of generation
        
        # === Voice Cloning / Reference Audio ===
        "ref_audio": "./elon_short.wav", # Path to your reference voice sample (can be a list for multiple)
        "ref_text": None,                # Exact transcript of the reference sample (None = Auto transcribe)
        
        # === Output & File Formatting ===
        "output_path": "./",             # Where to save the output files
        "file_prefix": f"elon_tts_{int(time.time())}", # Unique filename so it stops overwriting!
        "audio_format": "wav",           # Output format ("wav", "flac")
        "join_audio": True,              # If True, joins all chunks into one audio file instead of multiple
        
        # === Model Specific Tweaks (Optional) ===
        "cfg_scale": None,               # Classifier-free guidance scale
        "ddpm_steps": None,              # Override diffusion steps
        "prompt": None,                  # Optional model-specific prompt prefix
        "instruct": "excited",           # Style/emotion instruction!

        
        # === Playback & Streaming ===
        "play": False,                   # Play the audio locally as it generates
        "stream": False,                 # Stream audio in segments during generation
        "save": False,                   # Used with stream=True to save the segments
        "streaming_interval": 2.0,       # Time interval in seconds for streaming
        
        # === Logging ===
        "verbose": False                 # Set to False to hide the messy iteration progress logs!
    }

    print("\n[+] Setting up TTS generator...")
    print(f"[-] Model: {tts_kwargs['model']}")
    print(f"[-] Text: {tts_kwargs['text']}")
    print("[-] Verbose mode is OFF. The messy logs are hidden.")

    start_time = time.time()
    
    # Run the generator
    generate_audio(**tts_kwargs)
    
    end_time = time.time()
    
    output_filename = f"{tts_kwargs['file_prefix']}.{tts_kwargs['audio_format']}"

    print(f"\n[+] Generation completed in {end_time - start_time:.2f} seconds.")
    print(f"[+] Look for {output_filename} in your directory!")

if __name__ == "__main__":
    main()
