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

    from dummy_test import DummyTest
    dummy = DummyTest(asr)
    print(dummy.test("ravdess"))
    print(dummy.test("icmocap"))

# Correct: 286 Total: 672 Accuracy: 42.560%
# label vs prediction {'neu': Counter({'fru': 61, 'ang': 30, 'sad': 3, 'neu': 2}), 'ang': Counter({'ang': 190, 'fru': 2}), 'sad': Counter({'ang': 72, 'fru': 69, 'exc': 24, 'sad': 19, 'neu': 4, 'hap': 4}), 'hap': Counter({'ang': 154, 'fru': 27, 'exc': 10, 'hap': 1})}
# Correct: 2458 Total: 4490 Accuracy: 54.744%
# label vs prediction {'neu': Counter({'neu': 1134, 'fru': 261, 'sad': 250, 'exc': 43, 'ang': 13, 'hap': 7}), 'ang': Counter({'ang': 927, 'fru': 130, 'exc': 28, 'neu': 9, 'sad': 8, 'hap': 1}), 'sad': Counter({'sad': 981, 'fru': 51, 'neu': 31, 'hap': 9, 'exc': 8, 'ang': 4}), 'hap': Counter({'exc': 191, 'sad': 153, 'hap': 120, 'neu': 105, 'fru': 19, 'ang': 7})}