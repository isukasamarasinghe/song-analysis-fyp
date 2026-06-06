import numpy as np

def recognise_chords(chroma, sr=22050, hop_length=512):
    
    print('Running chord recognition...')
    
    # Chord templates for major and minor chords
    chord_templates = {}
    
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 
             'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Major chord intervals: root, major third, perfect fifth
    major_intervals = [0, 4, 7]
    
    # Minor chord intervals: root, minor third, perfect fifth
    minor_intervals = [0, 3, 7]
    
    # Build templates for all 12 major and minor chords
    for i, note in enumerate(notes):
        # Major template
        major_template = np.zeros(12)
        for interval in major_intervals:
            major_template[(i + interval) % 12] = 1
        chord_templates[f'{note} Major'] = major_template
        
        # Minor template
        minor_template = np.zeros(12)
        for interval in minor_intervals:
            minor_template[(i + interval) % 12] = 1
        chord_templates[f'{note} Minor'] = minor_template
    
    # Detect chord for each frame
    frame_chords = []
    for frame in range(chroma.shape[1]):
        frame_chroma = chroma[:, frame]
        
        best_chord = 'N/A'
        best_score = -1
        
        for chord_name, template in chord_templates.items():
            score = np.dot(frame_chroma, template) / (
                np.linalg.norm(frame_chroma) * np.linalg.norm(template) + 1e-6
            )
            if score > best_score:
                best_score = score
                best_chord = chord_name
        
        frame_chords.append(best_chord)
    
    # Group consecutive same chords together with timestamps
    grouped_chords = []
    current_chord = frame_chords[0]
    start_frame = 0
    
    for i in range(1, len(frame_chords)):
        if frame_chords[i] != current_chord:
            start_time = round(start_frame * hop_length / sr, 2)
            end_time = round(i * hop_length / sr, 2)
            
            grouped_chords.append({
                'chord': current_chord,
                'start': start_time,
                'end': end_time,
                'start_formatted': format_time(start_time),
                'end_formatted': format_time(end_time)
            })
            
            current_chord = frame_chords[i]
            start_frame = i
    
    # Simplify - only keep chords longer than 1 second
    simplified = [c for c in grouped_chords if (c['end'] - c['start']) >= 0.3]
    
    print(f'Detected {len(simplified)} chord changes')
    return simplified

def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f'{mins}:{secs:02d}'