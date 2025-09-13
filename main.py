import torch
import pyaudio
import wave
import numpy as np
import os
import tempfile
import time
import warnings
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline

warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*return_token_timestamps.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*return_token_timestamps.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*return_token_timestamps.*")

def record_audio(duration=5, sample_rate=16000):
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
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_filename = temp_file.name
        temp_file.close()
        
        wf = wave.open(temp_filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(actual_rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        return temp_filename
        
    except Exception as e:
        print(f"Error recording audio: {e}")
        return None
    finally:
        p.terminate()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-small"

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

print("Starting microphone recording for multilingual speech recognition (Hindi/English)...")
audio_file = record_audio(duration=5)

if audio_file:
    try:
        result = pipe(audio_file)
        print(f"Transcription: {result['text']}")
    except Exception as e:
        print(f"Error processing audio: {e}")
    finally:
        if os.path.exists(audio_file):
            os.remove(audio_file)
            print("Temporary audio file cleaned up.")
else:
    print("Failed to record audio. Please check your microphone connection.")