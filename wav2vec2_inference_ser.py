import torch
import librosa
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import torch.nn as nn

# Improvements: 
# - inference time log

class Wave2Vec2InferenceSER:
    def __init__(self,model_name, hotwords=[], use_lm_if_possible=True, use_gpu=True):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

        self.model.projector = nn.Linear(1024, 1024, bias=True)
        self.model.classifier = nn.Linear(1024, 8, bias=True)

        torch_state_dict = torch.load('pytorch_model.bin', map_location="cpu")

        self.model.projector.weight.data = torch_state_dict['classifier.dense.weight']
        self.model.projector.bias.data = torch_state_dict['classifier.dense.bias']

        self.model.classifier.weight.data = torch_state_dict['classifier.output.weight']
        self.model.classifier.bias.data = torch_state_dict['classifier.output.bias']

        self.model.to(self.device)

    def buffer_to_text(self, audio_buffer):
        if len(audio_buffer) == 0:
            return ""

        inputs = self.processor(torch.tensor(audio_buffer), sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device),
                                attention_mask=inputs.attention_mask.to(self.device)).logits
        
        id2label = {
                "0": "angry",
                "1": "calm",
                "2": "disgust",
                "3": "fearful",
                "4": "happy",
                "5": "neutral",
                "6": "sad",
                "7": "surprised"
            }
        interp = id2label[str(torch.argmax(logits, dim=-1).cpu().numpy()[0])]
        return interp

    def file_to_text(self, filename):
        audio_input, samplerate = librosa.load(filename, sr=16000)
        assert samplerate == 16000
        return self.buffer_to_text(audio_input)
    
if __name__ == "__main__":
    print("Model test")
    asr = Wave2Vec2InferenceSER("jonatasgrosman/wav2vec2-large-english")
    interp = asr.file_to_text("../Audio_Speech_Actors_01-24/03-01-08-02-01-01-01.wav")
    print(interp)