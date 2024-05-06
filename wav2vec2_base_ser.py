import torch
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Improvements: 
# - inference time log

class Wav2Vec2InferenceSER:
    def __init__(self,model_name, hotwords=[], use_lm_if_possible=True, use_gpu=True):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = AutoModelForAudioClassification.from_pretrained("canlinzhang/wav2vec2_speech_emotion_recognition_trained_on_IEMOCAP").to(self.device)
        self.processor = AutoFeatureExtractor.from_pretrained("canlinzhang/wav2vec2_speech_emotion_recognition_trained_on_IEMOCAP")


    def buffer_to_text(self, audio_buffer):
        if len(audio_buffer) == 0:
            return ""

        inputs = self.processor(audio_buffer, sampling_rate=16000, return_tensors="pt", padding="longest")
        with torch.no_grad():
            pred = self.model(inputs.input_values.to(self.device)).logits
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
    asr = Wav2Vec2InferenceSER("")
    interp = asr.file_to_text("C:/temp/Audio_Speech_Actors_01-24/03-01-05-02-01-02-08.wav")
    print(interp)

    from dummy_test import DummyTest
    dummy = DummyTest(asr)
    print(dummy.test("ravdess"))
    print(dummy.test("icmocap"))

# Correct: 230 Total: 672 Accuracy: 34.226%
# label vs prediction {'neu': Counter({'ang': 87, 'hap': 5, 'neu': 4}), 'ang': Counter({'ang': 192}), 'sad': Counter({'ang': 134, 'hap': 45, 'neu': 11, 'sad': 2}), 'hap': Counter({'ang': 160, 'hap': 32})}
# Correct: 4015 Total: 4490 Accuracy: 89.421%
# {'neu': Counter({'neu': 1503, 'sad': 97, 'hap': 80, 'ang': 28}), 'ang': Counter({'ang': 1052, 'neu': 21, 'sad': 16, 'hap': 14}), 'sad': Counter({'sad': 967, 'hap': 56, 'neu': 55, 'ang': 6}), 'hap': Counter({'hap': 493, 'neu': 70, 'sad': 29, 'ang': 3})}