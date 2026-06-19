import os
import sys
import time
import glob
from mlx_audio.tts.generate import generate_audio
import mlx.core as mx

def combine_wavs(input_pattern, output_filename):
    from mlx_audio.audio_io import read, write
    import numpy as np
    
    files = sorted(glob.glob(input_pattern))
    if not files:
        return
        
    chunks = []
    sample_rate = None
    for f in files:
        audio, sr = read(f)
        chunks.append(audio)
        sample_rate = sr
        
    combined = np.concatenate(chunks, axis=0)
    write(output_filename, combined, sample_rate)
    
    # Cleanup individual chunks
    for f in files:
        os.remove(f)

def main():
    text = "My fellow Americans, today we face a choice not just about the policies we enact, but about the very soul of our nation. For too long, the hardworking men and women of this country have been forgotten, left behind by a system that rewards the few at the expense of the many. But no more. We are going to rebuild our cities, restore our borders, and ensure that every single citizen has the opportunity to achieve the American Dream. We will stand tall, we will stand proud, and together, we will forge a future that is brighter, stronger, and more prosperous than ever before. Thank you, and God bless you all."
    
    # Split text into sentences to prevent exponential memory ballooning
    sentences = [s.strip() + "." for s in text.split(".") if s.strip()]
    
    model_id = "mlx-community/MisoLabs-MisoTTS-bf16"
    print(f"[+] Starting sentence-by-sentence chunking workaround for 16-bit model...")
    print(f"[+] Total sentences to process: {len(sentences)}\n")
    
    start_time = time.time()
    
    for i, sentence in enumerate(sentences):
        print(f"[*] Processing Chunk {i+1}/{len(sentences)}: '{sentence}'")
        chunk_start = time.time()
        
        # Suppress stderr to hide the tqdm bar for each chunk
        original_stderr = sys.stderr
        with open(os.devnull, 'w') as devnull:
            sys.stderr = devnull
            generate_audio(
                text=sentence,
                model=model_id,
                voice="af_heart", # Standard high-quality voice
                output_path="./",
                file_prefix=f"temp_chunk_{i:03d}",
                audio_format="wav",
                verbose=False
            )
        sys.stderr = original_stderr
        
        # CLEAR MLX UNIFIED MEMORY CACHE TO PREVENT SWAPPING!
        mx.metal.clear_cache()
        
        print(f"    -> Chunk {i+1} finished in {time.time() - chunk_start:.2f}s")

    print("\n[+] Combining all chunks into final seamless audio file...")
    combine_wavs("temp_chunk_*.wav", "final_16bit_speech.wav")
    
    end_time = time.time()
    print(f"\n[+] SUCCESS! 16-bit generation completed using memory workaround in {end_time - start_time:.2f} seconds.")
    print(f"[+] Output saved as final_16bit_speech.wav")

if __name__ == "__main__":
    main()
