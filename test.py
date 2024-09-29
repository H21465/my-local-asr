import mlx_whisper
# import pyaudio
import audio2wav

audio2wav.record_audio('test.wav')

text = mlx_whisper.transcribe('test.wav', path_or_hf_repo="mlx-community/whisper-large-v3-mlx")
print(text["text"])