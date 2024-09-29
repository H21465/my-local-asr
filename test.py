import mlx_whisper
import pyaudio

text = mlx_whisper.transcribe('test.wav', path_or_hf_repo="mlx-community/whisper-large-v3-mlx")["text"]
print(text)
# test