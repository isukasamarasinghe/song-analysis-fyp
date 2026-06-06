import librosa
import numpy as np

def extract_features(file_path):
    print(f'Loading audio from: {file_path}')
    
    # Load audio file
    audio, sr = librosa.load(file_path, mono=True, sr=22050)
    
    # Get duration
    duration = librosa.get_duration(y=audio, sr=sr)
    
    # Extract chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    # Extract tempo and beats
    tempo_result, beats = librosa.beat.beat_track(y=audio, sr=sr)
    
    # Handle tempo whether it comes back as array or scalar
    if hasattr(tempo_result, '__len__'):
        tempo = float(tempo_result[0])
    else:
        tempo = float(tempo_result)
    
    # Chord transition frequency
    chroma_diff = np.diff(chroma, axis=1)
    chord_transition_freq = float(np.mean(np.abs(chroma_diff)))
    
    print(f'Duration: {duration:.2f} seconds')
    print(f'Tempo: {tempo:.2f} BPM')
    print(f'Chroma shape: {chroma.shape}')
    print(f'MFCC shape: {mfcc.shape}')
    
    return {
        'audio': audio,
        'sr': sr,
        'duration': float(duration),
        'chroma': chroma,
        'mfcc': mfcc,
        'tempo': tempo,
        'beats': beats,
        'chord_transition_freq': chord_transition_freq
    }