import torch
import librosa
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import torch.nn as nn

# Improvements: 
# - inference time log

class Wav2Vec2InferenceSER:
    def __init__(self,model_name, hotwords=[], use_lm_if_possible=True, use_gpu=True):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

        self.model.projector = nn.Linear(1024, 1024, bias=True)
        self.model.classifier = nn.Linear(1024, 8, bias=True)

        torch_state_dict = torch.load('pytorch_model.bin')

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
                0: "angry",
                1: "calm",
                2: "disgust",
                3: "fearful",
                4: "happy",
                5: "neutral",
                6: "sad",
                7: "surprised"
            }
        prob = torch.nn.functional.softmax(logits, dim=1)
        interp = id2label[torch.argmax(prob).item()]
        emotion_prob = {id2label[i]: "{:.3f}".format(v) for i, v in enumerate(prob.cpu().numpy()[0])}

        return interp, emotion_prob

    def file_to_text(self, filename):
        audio_input, samplerate = librosa.load(filename, sr=16000)
        assert samplerate == 16000
        return self.buffer_to_text(audio_input)
    
if __name__ == "__main__":
    print("Model test")
    asr = Wav2Vec2InferenceSER("")
    interp = asr.file_to_text("C:/temp/Audio_Speech_Actors_01-24/03-01-05-02-01-02-08.wav")
    print(interp)

    from dummy_test import DummyTest
    dummy = DummyTest(asr)
    print(dummy.test("ravdess"))
    print(dummy.test("icmocap"))

# Correct: 591 Total: 672 Accuracy: 87.946%
# label vs prediction {'neu': Counter({'neu': 75, 'cal': 17, 'hap': 3, 'ang': 1}), 'ang': Counter({'ang': 187, 'hap': 2, 'sur': 2, 'dis': 1}), 'sad': Counter({'sad': 155, 'cal': 16, 'fea': 8, 'dis': 7, 'hap': 4, 'neu': 2}), 'hap': Counter({'hap': 174, 'ang': 8, 'neu': 4, 'sur': 3, 'fea': 2, 'dis': 1})}
# Correct: 1198 Total: 4490 Accuracy: 26.682%
# label vs prediction {'neu': Counter({'hap': 468, 'sad': 429, 'sur': 342, 'cal': 196, 'neu': 115, 'fea': 70, 'dis': 55, 'ang': 33}), 'ang': Counter({'hap': 416, 'ang': 361, 'sur': 224, 'dis': 34, 'sad': 26, 'fea': 17, 'neu': 13, 'cal': 12}), 'sad': Counter({'sad': 582, 'cal': 233, 'hap': 119, 'fea': 63, 'neu': 36, 'sur': 36, 'dis': 13, 'ang': 2}), 'hap': Counter({'sad': 225, 'hap': 140, 'cal': 77, 'sur': 57, 'fea': 48, 'dis': 24, 'neu': 20, 'ang': 4})}