import os
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'songanalyser2025'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

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

    print(f'File received: {filename}')
    print(f'Instrument: {instrument}')
    print(f'Saved to: {filepath}')

    return f'File received: {filename} | Instrument: {instrument}'

if __name__ == '__main__':
    app.run(debug=True)