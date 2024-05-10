import torch
import librosa
from faster_whisper import WhisperModel
import torch.nn as nn

# Improvements: 
# - inference time log

class WhisperInference:
    def __init__(self, model_name, hotwords=[], use_lm_if_possible=True, use_gpu=True):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = WhisperModel("distil-large-v2", device="cuda", compute_type="float32")

    def buffer_to_text(self, audio_buffer):
        segments, info = self.model.transcribe(audio_buffer, beam_size=5)
        
        return [seg.text for seg in segments]
        
    def file_to_text(self, filename):
        audio_input, samplerate = librosa.load(filename, sr=16000)
        assert samplerate == 16000
        return self.buffer_to_text(audio_input)
    
if __name__ == "__main__":
    print("Model test")
    asr = WhisperInference("")
    interp = asr.file_to_text("../Audio_Speech_Actors_01-24/03-01-05-02-01-02-08.wav")
    print(interp)