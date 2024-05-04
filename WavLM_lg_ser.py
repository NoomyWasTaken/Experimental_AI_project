import torch
import librosa
from transformers import AutoModelForAudioClassification
import torch.nn as nn

# Improvements: 
# - inference time log

class WavLMInferenceSER:
    def __init__(self, model_name, hotwords=[], use_lm_if_possible=True, use_gpu=True):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = AutoModelForAudioClassification.from_pretrained("3loi/SER-Odyssey-Baseline-WavLM-Categorical-Attributes", trust_remote_code=True)
        self.model.to(self.device)

    def buffer_to_text(self, audio_buffer):
        if len(audio_buffer) == 0:
            return ""

        norm_wav = (audio_buffer - self.model.config.mean) / (self.model.config.std+0.000001)
        mask = torch.ones(1, len(norm_wav)).to(self.device)
        wavs = torch.tensor(norm_wav).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(wavs, mask)
        prob = torch.nn.functional.softmax(pred, dim=1)
        interp = self.model.config.id2label[torch.argmax(prob).item()]
        emotion_prob = {self.model.config.id2label[i]: "{:.3f}".format(v) for i, v in enumerate(prob.cpu().numpy()[0])}
        
        return interp, emotion_prob

    def file_to_text(self, filename):
        audio_input, samplerate = librosa.load(filename, sr=16000)
        assert samplerate == 16000
        return self.buffer_to_text(audio_input)
    
if __name__ == "__main__":
    print("Model test")
    asr = WavLMInferenceSER("")
    interp = asr.file_to_text("C:/temp/Audio_Speech_Actors_01-24/03-01-05-02-01-02-08.wav")
    print(interp)