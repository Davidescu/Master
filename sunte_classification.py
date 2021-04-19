import librosa 
audio_path =r'C:\Users\Daniel\Downloads\sunet1.mp4'
x , sr = librosa.load(audio_path)
print(type(x), type(sr))