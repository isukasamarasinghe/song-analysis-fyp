import librosa
import numpy as np

def segment_structure(mfcc, sr=22050, hop_length=512):
    
    print('Running structure segmentation...')
    
    # Compute recurrence matrix
    rec_matrix = librosa.segment.recurrence_matrix(
        mfcc,
        mode='affinity',
        sym=True
    )
    
    # Find structural boundaries
    bounds = librosa.segment.agglomerative(mfcc, 8)
    bound_times = librosa.frames_to_time(bounds, sr=sr, hop_length=hop_length)
    
    # Label each section
    section_labels = ['Intro', 'Verse', 'Chorus', 'Verse', 
                      'Chorus', 'Bridge', 'Chorus', 'Outro']
    
    sections = []
    for i in range(len(bound_times) - 1):
        label = section_labels[i] if i < len(section_labels) else f'Section {i+1}'
        start = round(float(bound_times[i]), 2)
        end = round(float(bound_times[i + 1]), 2)
        sections.append({
            'label': label,
            'start': start,
            'end': end,
            'start_formatted': format_time(start),
            'end_formatted': format_time(end)
        })
    
    print(f'Found {len(sections)} sections')
    return sections

def format_time(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f'{mins}:{secs:02d}'