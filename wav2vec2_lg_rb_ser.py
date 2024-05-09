import torch
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Improvements: 
# - inference time log

class Wav2Vec2InferenceSER:
    def __init__(self,model_name, hotwords=[], use_lm_if_possible=True, use_gpu=True):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model = AutoModelForAudioClassification.from_pretrained("../wav2vec2_finetune/saves/checkpoint_ft_wav2vec2_lg_rb_msp").to(self.device)
        self.processor = AutoFeatureExtractor.from_pretrained("../wav2vec2_finetune/saves/checkpoint_ft_wav2vec2_lg_rb_msp")


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

# Correct: 212 Total: 672 Accuracy: 31.548%
# label vs prediction {'neu': Counter({'fru': 61, 'ang': 30, 'sad': 3, 'neu': 2}), 'ang': Counter({'ang': 190, 'fru': 2}), 'sad': Counter({'ang': 72, 'fru': 69, 'exc': 24, 'sad': 19, 'neu': 4, 'hap': 4}), 'hap': Counter({'ang': 154, 'fru': 27, 'exc': 10, 'hap': 1})}
# Correct: 3162 Total: 4490 Accuracy: 70.423%
# label vs prediction {'neu': Counter({'neu': 1134, 'fru': 261, 'sad': 250, 'exc': 43, 'ang': 13, 'hap': 7}), 'ang': Counter({'ang': 927, 'fru': 130, 'exc': 28, 'neu': 9, 'sad': 8, 'hap': 1}), 'sad': Counter({'sad': 981, 'fru': 51, 'neu': 31, 'hap': 9, 'exc': 8, 'ang': 4}), 'hap': Counter({'exc': 191, 'sad': 153, 'hap': 120, 'neu': 105, 'fru': 19, 'ang': 7})}