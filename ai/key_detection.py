import numpy as np

def detect_key(chroma):
    
    # Major and minor key profiles by Krumhansl-Schmuckler
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F',
             'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Average chroma across the whole song
    chroma_mean = np.mean(chroma, axis=1)
    
    best_score = -2
    best_key = ''
    best_mode = ''
    
    for i in range(12):
        # Rotate chroma to match each key
        rotated = np.roll(chroma_mean, -i)
        
        # Compare against major profile
        major_score = np.corrcoef(rotated, major_profile)[0, 1]
        
        # Compare against minor profile
        minor_score = np.corrcoef(rotated, minor_profile)[0, 1]
        
        if major_score > best_score:
            best_score = major_score
            best_key = notes[i]
            best_mode = 'Major'
        
        if minor_score > best_score:
            best_score = minor_score
            best_key = notes[i]
            best_mode = 'Minor'
    
    return {
        'key': best_key,
        'mode': best_mode,
        'full_key': f'{best_key} {best_mode}',
        'confidence': round(float(best_score), 4)
    }