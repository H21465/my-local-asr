# # import sys
# # from PyQt6 import QtCore, QtWidgets

# # class MainWindow(QtWidgets.QWidget):
# # 	def __init__(self):
# # 		super(MainWindow, self).__init__()
# # 		self.setGeometry(300, 50, 400, 350)

# # 		self.setWindowFlags(
# # 			QtCore.Qt.WindowType.WindowStaysOnTopHint
# # 		)

# # 		self.label.setGeometry(50, 50, 100, 10)

# # if __name__ == '__main__':
# # 	app = QtWidgets.QApplication(sys.argv)
# # 	main_window = MainWindow()
# # 	main_window.show()
# # 	sys.exit(app.exec())

# import sys
# from PyQt6 import QtCore, QtWidgets

# class MainWindow(QtWidgets.QWidget):
#     def __init__(self):
#         super(MainWindow, self).__init__()
#         self.setGeometry(300, 50, 400, 350)

#         self.setWindowFlags(
#             QtCore.Qt.WindowType.WindowStaysOnTopHint
#         )

#         self.label = QtWidgets.QLabel(self)
#         self.label.setGeometry(50, 50, 300, 200)
#         self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)

#         self.texts = ["hello", "world", "こんにちは", "世界"]
#         self.current_index = 0

#         self.timer = QtCore.QTimer(self)
#         self.timer.timeout.connect(self.update_text)
#         self.timer.start(1000)  # 1000ミリ秒 = 1秒

#     def update_text(self):
#         if self.current_index < len(self.texts):
#             current_text = self.label.text()
#             new_text = current_text + self.texts[self.current_index] + "\n"
#             self.label.setText(new_text)
#             self.current_index += 1

# if __name__ == '__main__':
#     app = QtWidgets.QApplication(sys.argv)
#     main_window = MainWindow()
#     main_window.show()
#     sys.exit(app.exec())

import sys
import time
import queue
import threading
import numpy as np
import pyaudio
import mlx_whisper
from PyQt6 import QtCore, QtWidgets

class AudioRecorder:
    def __init__(self, rate=16000, chunk=1024, channels=1, record_seconds=3):
        self.rate = rate
        self.chunk = chunk
        self.channels = channels
        self.record_seconds = record_seconds
        self.format = pyaudio.paFloat32
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()

    def record_audio(self):
        pa = pyaudio.PyAudio()
        stream = pa.open(rate=self.rate,
                    channels=self.channels,
                    format=self.format,
                    input=True,
                    frames_per_buffer=self.chunk)

        while not self.stop_event.is_set():
            data = stream.read(self.chunk)
            self.audio_queue.put(np.frombuffer(data, dtype=np.float32))

        stream.stop_stream()
        stream.close()
        pa.terminate()

    def start_recording(self):
        self.stop_event.clear()
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.start()

    def stop_recording(self):
        self.stop_event.set()
        self.recording_thread.join()

    def get_audio_chunk(self):
        required_chunks = int(self.rate / self.chunk * self.record_seconds)
        audio_data = []

        while len(audio_data) < required_chunks:
            try:
                audio_data.append(self.audio_queue.get(timeout=1))
            except queue.Empty:
                if self.stop_event.is_set():
                    break

        return np.concatenate(audio_data) if audio_data else None

class MainWindow(QtWidgets.QWidget):
    update_text_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setGeometry(300, 50, 400, 350)
        self.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint)

        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(50, 50, 300, 200)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        self.label.setWordWrap(True)

        self.update_text_signal.connect(self.update_text)

        self.recorder = AudioRecorder()
        self.recorder.start_recording()

        self.transcribe_thread = threading.Thread(target=self.transcribe_audio)
        self.transcribe_thread.daemon = True
        self.transcribe_thread.start()

    def update_text(self, text):
        current_text = self.label.text()
        new_text = current_text + text + "\n"
        self.label.setText(new_text)

    def transcribe_audio(self):
        while True:
            frame = self.recorder.get_audio_chunk()
            if frame is not None:
                text = mlx_whisper.transcribe(frame, path_or_hf_repo="mlx-community/whisper-large-v3-turbo")
                if text["text"].strip() and text["text"] != "ご視聴ありがとうございました" and text["text"] != "おやすみなさい" and text["text"] != "ありがとうございました":
                    self.update_text_signal.emit(text["text"])

    def closeEvent(self, event):
        self.recorder.stop_recording()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())