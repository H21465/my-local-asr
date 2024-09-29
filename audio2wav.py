import pyaudio
import numpy as np

def record_audio(duration=3):
	RATE = 16000
	CHUNK = 1024
	FORMAT = pyaudio.paFloat32
	CHANNELS = 1

	pa = pyaudio.PyAudio()
	stream = pa.open(rate = RATE,
			channels = CHANNELS,
			format = FORMAT,
			input = True,
			frames_per_buffer = CHUNK)

	frames = []
	for _ in range(0, int(RATE / CHUNK * duration)):
		data = stream.read(CHUNK)
		frames.append(np.frombuffer(data, dtype=np.float32))
	
	stream.stop_stream()
	stream.close()
	pa.terminate()
	
	return np.concatenate(frames)

# record_audio("test.wav")
