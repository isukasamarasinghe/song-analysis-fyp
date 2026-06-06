from ai.feature_extraction import extract_features
from ai.key_detection import detect_key
from ai.structure_segmentation import segment_structure
from ai.chord_recognition import recognise_chords
from ai.difficulty_classifier import train_classifier, classify_difficulty

# Train the model first
train_classifier()

# Extract features
result = extract_features('uploads/test.mp3')
key_result = detect_key(result['chroma'])
sections = segment_structure(result['mfcc'])
chords = recognise_chords(result['chroma'])

# Build difficulty features
difficulty_features = {
    'unique_chord_count': len(set([c['chord'] for c in chords])),
    'chord_transition_freq': result['chord_transition_freq'],
    'tempo': result['tempo'],
    'num_sections': len(sections)
}

difficulty_result = classify_difficulty(difficulty_features)

print('\n--- FULL ANALYSIS RESULTS ---')
print(f"Duration: {result['duration']:.2f} seconds")
print(f"Tempo: {result['tempo']:.2f} BPM")
print(f"Key: {key_result['full_key']}")
print(f"Difficulty: {difficulty_result['difficulty']}")
print(f"Difficulty Confidence: {difficulty_result['confidence']}%")

print(f"\nSong Structure:")
for section in sections:
    print(f"  {section['label']}: {section['start_formatted']} - {section['end_formatted']}")

print(f"\nFirst 10 Chords:")
for chord in chords[:10]:
    print(f"  {chord['start_formatted']} - {chord['end_formatted']}: {chord['chord']}")