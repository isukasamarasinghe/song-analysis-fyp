from ai.feature_extraction import extract_features

result = extract_features('uploads/test.mp3')

print('\n--- RESULTS ---')
print(f"Duration: {result['duration']} seconds")
print(f"Tempo: {result['tempo']} BPM")
print(f"Chroma shape: {result['chroma'].shape}")
print(f"MFCC shape: {result['mfcc'].shape}")
print(f"Chord transition frequency: {result['chord_transition_freq']}")