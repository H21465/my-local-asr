import pyaudio
import numpy as np
import scipy.io.wavfile

def record_audio(output_name):
	RATE = 12000
	CHUNK = 1024
	FORMAT = pyaudio.paFloat32
	CHANNELS = 1

	pa = pyaudio.PyAudio()
	stream = pa.open(rate = RATE,
			channels = CHANNELS,
			format = FORMAT,
			input = True,
			frames_per_buffer = CHUNK)

	print("RECORD START")
	print("ctrl + c : STOP RECORDING")

	frame = []
	while True:
		try:
			d = stream.read(CHUNK)
			d = np.frombuffer(d, dtype=np.float32)
			frame.append(d)
		except KeyboardInterrupt:
			break

	stream.stop_stream()
	stream.close()
	pa.terminate()

	frame = np.array(frame).flatten()
	print("STOP {} Samples {:.2f}s".format(frame.size, frame.size/RATE))
	scipy.io.wavfile.write(output_name, RATE, frame)

# record_audio("test.wav")
