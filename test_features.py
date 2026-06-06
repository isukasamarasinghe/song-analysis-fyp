from ai.feature_extraction import extract_features
from ai.key_detection import detect_key
from ai.structure_segmentation import segment_structure

result = extract_features('uploads/test.mp3')
key_result = detect_key(result['chroma'])
sections = segment_structure(result['mfcc'])

print('\n--- RESULTS ---')
print(f"Duration: {result['duration']:.2f} seconds")
print(f"Tempo: {result['tempo']:.2f} BPM")
print(f"Key: {key_result['full_key']}")
print(f"Confidence: {key_result['confidence']}")
print(f"\nSong Structure:")
for section in sections:
    print(f"  {section['label']}: {section['start_formatted']} - {section['end_formatted']}")