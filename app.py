import os
from flask import Flask, render_template, request, redirect, url_for, flash
from ai.feature_extraction import extract_features
from ai.key_detection import detect_key
from ai.structure_segmentation import segment_structure
from ai.chord_recognition import recognise_chords
from ai.difficulty_classifier import train_classifier, classify_difficulty

app = Flask(__name__)
app.secret_key = 'songanalyser2025'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Train classifier on startup
print('Training difficulty classifier...')
train_classifier()
print('Classifier ready.')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyse', methods=['POST'])
def analyse():
    if 'audio' not in request.files:
        flash('No file uploaded.')
        return redirect(url_for('home'))

    file = request.files['audio']
    instrument = request.form.get('instrument')

    if file.filename == '':
        flash('No file selected.')
        return redirect(url_for('home'))

    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload MP3 or WAV only.')
        return redirect(url_for('home'))

    # Save the file
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    print(f'Analysing: {filename}')

    # Run full AI pipeline
    features = extract_features(filepath)
    key_result = detect_key(features['chroma'])
    sections = segment_structure(features['mfcc'])
    chords = recognise_chords(features['chroma'])

    # Build difficulty features
    difficulty_features = {
        'unique_chord_count': len(set([c['chord'] for c in chords])),
        'chord_transition_freq': features['chord_transition_freq'],
        'tempo': features['tempo'],
        'num_sections': len(sections)
    }

    difficulty_result = classify_difficulty(difficulty_features)

    # Build results object
    results = {
        'filename': filename,
        'instrument': instrument,
        'duration': round(features['duration'], 2),
        'tempo': round(features['tempo'], 2),
        'key': key_result['full_key'],
        'key_confidence': key_result['confidence'],
        'difficulty': difficulty_result['difficulty'],
        'difficulty_confidence': difficulty_result['confidence'],
        'sections': sections,
        'chords': chords[:20],  # First 20 chords
        'total_chords': len(chords)
    }

    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)