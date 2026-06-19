import sys
import subprocess
import time

def main():
    text = (
        "My fellow citizens, we stand at a crossroads in our great nation's history. For too long, "
        "the voices of the hardworking people have been ignored by those in power. We are going to change that. "
        "We are going to bring back jobs to our communities, secure our borders, and ensure that our economy "
        "works for everyone, not just the elite few. The time for empty promises is over. It is time for action. "
        "We will rebuild our crumbling infrastructure, support our brave men and women in uniform, and protect "
        "the fundamental liberties that make this country exceptional. Together, we will forge a future of prosperity, "
        "strength, and unity. We will not back down, we will not surrender, and we will emerge stronger than ever before. "
        "Thank you, and God bless our great nation."
    )
    model_id = "mlx-community/MisoLabs-MisoTTS-8bit"
    output_prefix = "trump_tts"
    
    print(f"Loading {model_id} and generating audio natively via Metal...")
    
    cmd = [
        sys.executable, "-m", "mlx_audio.tts.generate",
        "--model", model_id,
        "--text", text,
        "--output_path", "./",
        "--file_prefix", output_prefix,
        # === VOICE CLONING OPTIONS ===
        "--ref_audio", "./trump.wav",
        "--ref_text", "Angered and appalled millions of Americans across the political spectrum"
    ]
    
    print("\nGenerating audio... Please wait.")
    print("(This process runs silently in the background and can take several minutes for long texts)")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode != 0:
        print("\nAn error occurred during generation:")
        print(result.stderr)
        return
    
    print(f"\nGeneration complete in {end_time - start_time:.2f} seconds!")
    print(f"Output saved as {output_prefix}.wav")
    
if __name__ == "__main__":
    main()
