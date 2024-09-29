import mlx_whisper
import audio2wav

frame = audio2wav.record_audio('test.wav')

# text = mlx_whisper.transcribe('test.wav', path_or_hf_repo="mlx-community/whisper-large-v3-mlx")
text = mlx_whisper.transcribe(frame, path_or_hf_repo="mlx-community/whisper-large-v3-mlx")
print(text["text"])