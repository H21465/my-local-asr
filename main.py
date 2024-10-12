import mlx_whisper
import time
import audio2wav
import threading
import queue

def record_audio_thread(audio_queue):
	while True:
		frame = audio2wav.record_audio()
		audio_queue.put(frame)

def transcribe_audio_thread(audio_queue):
	while True:
		frame = audio_queue.get()
		text = mlx_whisper.transcribe(frame, path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language="ja")
		if text["text"].strip() and text["text"] != "ご視聴ありがとうございました" and text["text"] != "おやすみなさい" and text["text"] != "ありがとうございました":
			print(text["text"])
		audio_queue.task_done()

def main():
	audio2wav.initialize_recorder()

	audio_queue = queue.Queue()

	record_thread = threading.Thread(target=record_audio_thread, args=(audio_queue,))
	transcribe_thread = threading.Thread(target=transcribe_audio_thread, args=(audio_queue,))

	record_thread.daemon = True
	transcribe_thread.daemon = True

	record_thread.start()
	transcribe_thread.start()

	try:
		while True:
			time.sleep(1)
	except KeyboardInterrupt:
		print("Recording stopped by user.")
	finally:
		audio2wav.cleanup()

if __name__ == "__main__":
	main()
