import torch
import pyaudio
import wave
import numpy as np
import os
import tempfile
import time
import warnings
from datetime import datetime
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*return_token_timestamps.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*return_token_timestamps.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*return_token_timestamps.*")

def record_audio(duration=5, sample_rate=16000):
    audio_folder = "audio"
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)
        print(f"Created audio folder: {audio_folder}")
    
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    
    p = pyaudio.PyAudio()
    
    sample_rates = [sample_rate, 44100, 48000, 22050, 11025]
    stream = None
    actual_rate = None
    
    print(f"Recording for {duration} seconds...")
    print("3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("Speak now!")
    
    for rate in sample_rates:
        try:
            stream = p.open(
                format=format,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=chunk
            )
            actual_rate = rate
            print(f"Using sample rate: {rate} Hz")
            break
        except Exception as e:
            if rate == sample_rates[-1]:
                print(f"Error recording audio: {e}")
                return None
            continue
    
    try:
        frames = []
        for i in range(0, int(actual_rate / chunk * duration)):
            data = stream.read(chunk)
            frames.append(data)
        
        print("Recording finished!")
        
        stream.stop_stream()
        stream.close()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = os.path.join(audio_folder, f"recording_{timestamp}.wav")
        
        wf = wave.open(audio_filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(actual_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"Audio saved to: {audio_filename}")
        return audio_filename
        
    except Exception as e:
        print(f"Error recording audio: {e}")
        return None
    finally:
        p.terminate()

def load_model_with_timing():
    print("Loading Oriserve/Whisper-Hindi2Hinglish-Swift model...")
    start_time = time.time()
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model_id = "Oriserve/Whisper-Hindi2Hinglish-Swift"
    
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(device)
        
        processor = WhisperProcessor.from_pretrained(model_id)
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device
        )
        
        load_time = time.time() - start_time
        print(f"Model loaded successfully in {load_time:.2f} seconds")
        print(f"Using device: {device}")
        print(f"Using dtype: {torch_dtype}")
        
        return pipe, load_time
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def transcribe_audio_with_timing(pipe, audio_file):
    print("Starting transcription...")
    start_time = time.time()
    
    try:
        result = pipe(audio_file)
        inference_time = time.time() - start_time
        
        print(f"Transcription completed in {inference_time:.2f} seconds")
        print(f"Transcription: {result['text']}")
        
        return result, inference_time
        
    except Exception as e:
        inference_time = time.time() - start_time
        print(f"Error during transcription after {inference_time:.2f} seconds: {e}")
        return None, inference_time

def save_inference_time(model_name, load_time, inference_time, audio_file, transcription):
    safe_model_name = model_name.replace("/", "_").replace("-", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_filename = f"{safe_model_name}_{timestamp}.txt"
    
    content = f"""Model: {model_name}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Audio File: {audio_file}
Transcription: {transcription}

Timing Details:
- Model Loading Time: {load_time:.2f} seconds
- Inference Time: {inference_time:.2f} seconds
- Total Processing Time: {load_time + inference_time:.2f} seconds

Device: {"CUDA" if torch.cuda.is_available() else "CPU"}
"""
    
    # Write to file
    try:
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Inference time saved to: {txt_filename}")
    except Exception as e:
        print(f"Error saving inference time: {e}")

def main():
    print("=== Hinglish Speech Recognition with Timing ===")
    print("Model: Oriserve/Whisper-Hindi2Hinglish-Swift")
    print("=" * 50)
    
    model_name = "Oriserve/Whisper-Hindi2Hinglish-Swift"
    
    pipe, load_time = load_model_with_timing()
    if pipe is None:
        print("Failed to load model. Exiting.")
        return
    
    print("\nStarting microphone recording for Hinglish speech recognition...")
    audio_file = record_audio(duration=5)
    
    if audio_file:
        try:
            result, inference_time = transcribe_audio_with_timing(pipe, audio_file)
            
            if result:
                print("\n" + "=" * 50)
                print("SUMMARY:")
                print(f"Model loading time: {load_time:.2f} seconds")
                print(f"Inference time: {inference_time:.2f} seconds")
                print(f"Total processing time: {load_time + inference_time:.2f} seconds")
                print("=" * 50)
                
                save_inference_time(
                    model_name, 
                    load_time, 
                    inference_time, 
                    audio_file, 
                    result['text']
                )
            
        except Exception as e:
            print(f"Error processing audio: {e}")
    else:
        print("Failed to record audio. Please check your microphone connection.")

if __name__ == "__main__":
    main()
