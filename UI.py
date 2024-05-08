# this is the UI for voice emotion detection
import tkinter as tk
import tkinter.scrolledtext as ScrolledText
from live_vad_asr import main
from live_vad_asr import DEFAULT_SAMPLE_RATE
import threading
import queue

#DEFAULT_SAMPLE_RATE = 16000

class RecorderApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Audio Recorder")
        self.queue = queue.Queue() # message queue

        # create start button
        self.start_btn = tk.Button(master, text="Start Recording", command=self.start_recording)
        self.start_btn.pack(pady=20)

        # create stop button
        self.stop_btn = tk.Button(master, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_btn.pack(pady=20)

        # create a textbox used to receive messages
        self.text_box = ScrolledText.ScrolledText(master, height=10, width=50)
        self.text_box.pack(pady=20)

        # this is an event that used to control the audio record thread stop.
        self.stop_event = threading.Event()

        # define color of each emotion:
        self.text_box.tag_config("bold", font=('Helvetica', '10', 'bold'))
        self.text_box.tag_config("Neutral", foreground="gray")
        self.text_box.tag_config("Happy", foreground="green")
        self.text_box.tag_config("Sad", foreground="blue")
        self.text_box.tag_config("Angry", foreground="red")
        self.text_box.tag_config("Surprise", foreground="orange")
        self.text_box.tag_config("Fear", foreground="purple")
        self.text_box.tag_config("Disgust", foreground="brown")
        self.text_box.tag_config("Contempt", foreground="cyan")

        # init text update loop
        self.update_textbox()

    def main_record_logic(self):

        import argparse
        parser = argparse.ArgumentParser(
            description="Stream from microphone to webRTC and silero VAD")

        parser.add_argument('-v', '--webRTC_aggressiveness', type=int, default=3,
                            help="Set aggressiveness of webRTC: an integer between 0 and 3, 0 being the least aggressive about filtering out non-speech, 3 the most aggressive. Default: 3")
        parser.add_argument('--nospinner', action='store_true',
                            help="Disable spinner")
        parser.add_argument('-d', '--device', type=int, default=None,
                            help="Device input index (Int) as listed by pyaudio.PyAudio.get_device_info_by_index(). If not provided, falls back to PyAudio.get_default_device().")

        parser.add_argument('-name', '--silaro_model_name', type=str, default="silero_vad",
                            help="select the name of the model. You can select between 'silero_vad',''silero_vad_micro','silero_vad_micro_8k','silero_vad_mini','silero_vad_mini_8k'")
        parser.add_argument('--reload', action='store_true',
                            help="download the last version of the silero vad")

        parser.add_argument('-ts', '--trig_sum', type=float, default=0.25,
                            help="overlapping windows are used for each audio chunk, trig sum defines average probability among those windows for switching into triggered state (speech state)")

        parser.add_argument('-nts', '--neg_trig_sum', type=float, default=0.07,
                            help="same as trig_sum, but for switching from triggered to non-triggered state (non-speech)")

        parser.add_argument('-N', '--num_steps', type=int, default=8,
                            help="nubmer of overlapping windows to split audio chunk into (we recommend 4 or 8)")

        parser.add_argument('-nspw', '--num_samples_per_window', type=int, default=4000,
                            help="number of samples in each window, our models were trained using 4000 samples (250 ms) per window, so this is preferable value (lesser values reduce quality)")

        parser.add_argument('-msps', '--min_speech_samples', type=int, default=10000,
                            help="minimum speech chunk duration in samples")

        parser.add_argument('-msis', '--min_silence_samples', type=int, default=500,
                            help=" minimum silence duration in samples between to separate speech chunks")
        ARGS = parser.parse_args()
        ARGS.rate = DEFAULT_SAMPLE_RATE
        main(ARGS,self.stop_event,self.queue)

    def start_recording(self):
        self.stop_event.clear()  # make sure this event is clear
        self.stop_btn.config(state=tk.NORMAL)
        self.start_btn.config(state=tk.DISABLED)  # disable start button
        self.queue.put("Recording started...")
        self.record_thread = threading.Thread(target=self.main_record_logic)
        self.record_thread.start()

    def stop_recording(self):
        self.stop_event.set() # set up the stop event
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)# disable stop button
        self.queue.put("Recording stopped.")
        self.record_thread.join()

    def update_textbox(self):
        try:
            while not self.queue.empty():
                message = self.queue.get_nowait()  # get message
                if message.startswith("text:"):
                    _, text = message.split("text:", 1)
                    self.text_box.insert(tk.END, "text: ", "bold")
                    self.text_box.insert(tk.END, text.strip() + '\n')
                    self.text_box.see(tk.END)
                elif message.startswith("emotion:"):
                    _, data = message.split("emotion:", 1)
                    emotion, probs = eval(data)
                    self.text_box.insert(tk.END, "emotion: ", "bold")
                    self.text_box.insert(tk.END, f"{emotion}", emotion)
                    #self.text_box.insert(tk.END, "\n")
                    #for emo, prob in probs.items():
                    #    self.text_box.insert(tk.END, f"{emo}: {prob} \n ", emo)
                    self.text_box.insert(tk.END, '\n')
                    self.text_box.see(tk.END)
                else:
                    self.text_box.insert(tk.END, message + '\n')  # update textbox
                    self.text_box.see(tk.END)  # roll to the latest message
        finally:
            self.master.after(100, self.update_textbox)  # run this function itself every 100 ms


if __name__ == "__main__":
    root = tk.Tk()
    app = RecorderApp(root)
    root.mainloop()
