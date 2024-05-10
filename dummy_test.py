import os, librosa
from torchaudio.datasets import IEMOCAP
from collections import Counter

class DummyTest():
    def __init__(self, model):
        self.model = model
        self.ravdess = self.load_ravdess("C:/temp/Audio_Speech_Actors_01-24")
        self.icmocap = self.load_icmocap("C:/temp/IEMOCAP_full_release_withoutVideos/IEMOCAP_full_release_withoutVideos")

    def load_ravdess(self, dataset_path):
        file_list = []
        id2label = {0: "neutral", 1: "calm", 2: "happy", 3: "sad", 4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"}
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".wav"):
                    label = int(file.split("-")[2])-1
                    temp = {"path": os.path.join(root, file), "label": id2label[label], "label_id": label}
                    file_list.append(temp)
        
        filter = {"neutral": 0, "angry": 1, "sad": 2, "happy": 3}
        file_list = [x for x in file_list if x["label"] in filter]
        for i in range(len(file_list)):
            file_list[i]["label_id"] = filter[file_list[i]["label"]]
    
        print(f"Filtered files in ravdess: {len(file_list)}")
        return file_list
    
    def load_icmocap(self, dataset_path):
        file_list = []
        dataset = IEMOCAP(dataset_path)
        folder = 'IEMOCAP'
        label2id = {"neu": 0, "hap": 1, "ang": 2, "sad": 3, "exc": 4, "fru": 5}
        filter = {"neu": 0, "ang": 1, "sad": 2, "hap": 3}
        for i in range(len(dataset)):
            file, _, _, label, _ = dataset.get_metadata(i)
            if label in filter:
                file_path = os.path.join(dataset_path, folder, file)
                temp = {"path": file_path, "label": label, "label_id": filter[label]}
                file_list.append(temp)
        
        print(f"Filtered files in icmocap: {len(file_list)}")
        return file_list

    
    def test(self, dataset):
        correct = 0
        total = 0
        count = {"neu": Counter(), "ang": Counter(), "sad": Counter(), "hap": Counter()}
        if dataset == "ravdess":
            for file in self.ravdess:
                interp, emotion_prob = self.model.file_to_text(file["path"])
                # print(interp, file["label"])
                if interp.lower()[:3] == file["label"][:3]:
                    correct += 1
                total += 1
                count[file["label"][:3]][interp.lower()[:3]] += 1   
            print("ravdess")
        elif dataset == "icmocap":
            for file in self.icmocap:
                interp, emotion_prob = self.model.file_to_text(file["path"])
                # print(interp, file["label"])
                if interp.lower()[:3] == file["label"][:3]:
                    correct += 1
                total += 1
                count[file["label"][:3]][interp.lower()[:3]] += 1
            print("icmocap")
        print(f"Correct: {correct} Total: {total} Accuracy: {correct/total*100:.3f}%")
        return count

if __name__ == "__main__":
    print("Dummy test")
    dummy = DummyTest("")
    print(dummy.ravdess[0])
    print(dummy.icmocap[0])