import sys
import os
import subprocess
import numpy as np
import sounddevice as sd
import soundfile as sf
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog, QVBoxLayout,
    QHBoxLayout, QProgressBar, QMessageBox, QComboBox, QGroupBox, QGridLayout,
    QTextEdit, QStyleFactory
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QIcon, QFont, QPixmap
import pyqtgraph as pg
import time
from pydub import AudioSegment
import datetime
import shutil
import requests
import re

class AudioRecorderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.is_recording = False
        self.is_playing = False
        self.audio_data = []
        self.fs = 44100  # High quality sample rate for recording
        self.channels = 1  # Mono audio
        self.input_device = None
        self.audio_file = 'recorded_audio.wav'
        self.waveform_data = np.zeros(44100)  # Initialize with one second of zeros
        self.chunk_size = 1024  # Number of samples per chunk

    def init_ui(self):
        # Set application style
        QApplication.setStyle(QStyleFactory.create('Fusion'))
        self.setStyleSheet("""
            QWidget {
                background-color: #000000;
                font-family: Arial;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #c7c7c7;
                color: #6d6d6d;
            }
            QLabel {
                font-size: 14px;
            }
            QComboBox {
                padding: 5px;
            }
            QProgressBar {
                border: 1px solid #bbb;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 20px;
            }
            QTextEdit {
                background-color: #000000;
                border: 1px solid #bbb;
                border-radius: 5px;
                padding: 5px;
            }
        """)

        # Fonts
        title_font = QFont("Arial", 16, QFont.Bold)
        label_font = QFont("Arial", 12)

        # Logo
        logo_label = QLabel()
        pixmap = QPixmap('logo.png')  # Make sure to have a logo.png in the working directory
        logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignCenter)

        # Recording Controls Group
        recording_group = QGroupBox("Recording Controls")
        recording_layout = QGridLayout()

        self.record_button = QPushButton('Record')
        self.stop_button = QPushButton('Stop')
        self.stop_button.setEnabled(False)

        recording_layout.addWidget(self.record_button, 0, 0)
        recording_layout.addWidget(self.stop_button, 0, 1)
        recording_layout.addWidget(QLabel('Progress:'), 1, 0, 1, 2)
        self.recording_progress = QProgressBar()
        recording_layout.addWidget(self.recording_progress, 2, 0, 1, 2)

        recording_group.setLayout(recording_layout)

        # Waveform Plot
        waveform_group = QGroupBox("Live Waveform")
        waveform_layout = QVBoxLayout()
        self.waveform_plot = pg.PlotWidget()
        self.waveform_plot.setYRange(-1, 1)
        self.waveform_curve = self.waveform_plot.plot(pen='g')
        waveform_layout.addWidget(self.waveform_plot)
        waveform_group.setLayout(waveform_layout)

        # Playback Controls Group
        playback_group = QGroupBox("Playback Controls")
        playback_layout = QGridLayout()

        self.play_button = QPushButton('Play')
        self.pause_button = QPushButton('Pause')
        self.pause_button.setEnabled(False)

        playback_layout.addWidget(self.play_button, 0, 0)
        playback_layout.addWidget(self.pause_button, 0, 1)
        playback_layout.addWidget(QLabel('Progress:'), 1, 0, 1, 2)
        self.playback_progress = QProgressBar()
        playback_layout.addWidget(self.playback_progress, 2, 0, 1, 2)

        playback_group.setLayout(playback_layout)

        # Action Buttons Group
        actions_group = QGroupBox("Actions")
        actions_layout = QGridLayout()

        self.export_button = QPushButton('Export')
        self.export_button.setEnabled(False)
        self.transcribe_button = QPushButton('Transcribe')
        self.transcribe_button.setEnabled(False)
        self.metadata_button = QPushButton('Generate Metadata')
        self.metadata_button.setEnabled(False)
        self.load_audio_button = QPushButton('Load Audio')

        actions_layout.addWidget(self.export_button, 0, 0)
        actions_layout.addWidget(self.transcribe_button, 0, 1)
        actions_layout.addWidget(self.metadata_button, 1, 0)
        actions_layout.addWidget(self.load_audio_button, 1, 1)

        actions_group.setLayout(actions_layout)

        # Device Selection Group
        device_group = QGroupBox("Input Device")
        device_layout = QVBoxLayout()
        self.device_combo = QComboBox()
        self.populate_audio_devices()
        device_layout.addWidget(self.device_combo)
        device_group.setLayout(device_layout)

        # Transcription and Metadata
        transcription_group = QGroupBox("Transcription")
        transcription_layout = QVBoxLayout()
        self.transcription_label = QTextEdit()
        self.transcription_label.setReadOnly(True)
        transcription_layout.addWidget(self.transcription_label)
        transcription_group.setLayout(transcription_layout)

        # Main Layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(logo_label)
        main_layout.addWidget(recording_group)
        main_layout.addWidget(waveform_group)
        main_layout.addWidget(playback_group)
        main_layout.addWidget(actions_group)
        main_layout.addWidget(device_group)
        main_layout.addWidget(transcription_group)

        self.setLayout(main_layout)
        self.setWindowTitle('Cassette to Digital Converter')
        self.setGeometry(100, 100, 800, 900)
        self.setWindowIcon(QIcon('icon.png'))  # Optional: Set your application icon

        # Connect signals to slots
        self.record_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        self.play_button.clicked.connect(self.start_playback)
        self.pause_button.clicked.connect(self.pause_playback)
        self.export_button.clicked.connect(self.export_audio)
        self.transcribe_button.clicked.connect(self.transcribe_audio)
        self.metadata_button.clicked.connect(self.generate_metadata)
        self.load_audio_button.clicked.connect(self.load_audio)

    def populate_audio_devices(self):
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                self.device_combo.addItem(f"{device['name']} (ID: {i})", i)

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.record_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.audio_data = []

            # Get the selected input device
            self.input_device = self.device_combo.currentData()

            def callback(indata, frames, time, status):
                if status:
                    print(status)
                self.audio_data.append(indata.copy())
                self.update_waveform(indata.copy())

            self.stream = sd.InputStream(
                samplerate=self.fs,
                channels=self.channels,
                callback=callback,
                blocksize=self.chunk_size,
                device=self.input_device  # Use the selected device
            )
            self.stream.start()

            # Start recording progress bar
            self.recording_progress.setValue(0)
            self.recording_timer = QTimer()
            self.recording_timer.timeout.connect(self.update_recording_progress)
            self.recording_timer.start(1000)
            self.recording_start_time = time.time()

    def update_waveform(self, new_data):
        # Update the waveform data with new samples
        self.waveform_data = np.roll(self.waveform_data, -len(new_data))
        self.waveform_data[-len(new_data):] = new_data.flatten()
        self.waveform_curve.setData(self.waveform_data)

    def update_recording_progress(self):
        elapsed = time.time() - self.recording_start_time
        # Max length for a C120 cassette is 120 minutes (7200 seconds)
        progress = min(int((elapsed / 7200) * 100), 100)
        self.recording_progress.setValue(progress)
        if progress >= 100:
            self.stop_recording()

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.stream.stop()
            self.stream.close()
            self.record_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.play_button.setEnabled(True)
            self.export_button.setEnabled(True)
            self.transcribe_button.setEnabled(True)
            self.recording_timer.stop()
            self.recording_progress.setValue(0)

            # Concatenate recorded data
            self.audio_data = np.concatenate(self.audio_data)
            sf.write(self.audio_file, self.audio_data, self.fs)
            QMessageBox.information(self, 'Recording Complete', 'Audio recording has been saved.')

    def start_playback(self):
        if not self.is_playing:
            self.is_playing = True
            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(True)

            def callback(outdata, frames, time, status):
                if status:
                    print(status)
                start = self.playback_pos
                end = start + frames
                if end > len(self.audio_data):
                    outdata[:len(self.audio_data[start:]), 0] = self.audio_data[start:].flatten()
                    outdata[len(self.audio_data[start:]):] = 0
                    self.stop_playback()
                    return
                outdata[:, 0] = self.audio_data[start:end].flatten()
                self.playback_pos = end

            self.playback_pos = 0
            self.play_stream = sd.OutputStream(
                samplerate=self.fs,
                channels=self.channels,
                callback=callback
            )
            self.play_stream.start()

            # Start playback progress bar
            self.playback_progress.setValue(0)
            self.playback_timer = QTimer()
            self.playback_timer.timeout.connect(self.update_playback_progress)
            self.playback_timer.start(1000)
            self.playback_start_time = time.time()

    def update_playback_progress(self):
        elapsed = self.playback_pos / self.fs
        total_duration = len(self.audio_data) / self.fs
        progress = min(int((elapsed / total_duration) * 100), 100)
        self.playback_progress.setValue(progress)

    def pause_playback(self):
        if self.is_playing:
            self.is_playing = False
            self.play_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.play_stream.stop()
            self.playback_timer.stop()

    def stop_playback(self):
        self.is_playing = False
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.play_stream.stop()
        self.playback_timer.stop()
        self.playback_progress.setValue(0)

    def export_audio(self):
        try:
            # Check if the audio file exists
            if not os.path.exists(self.audio_file):
                QMessageBox.warning(self, 'Export Error', 'Audio file not found.')
                return

            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Save Audio File", self.audio_file, "Audio Files (*.mp3 *.wav)", options=options
            )

            if file_name:
                # Copy the audio file to the new location
                shutil.copy2(self.audio_file, file_name)

                # Copy the transcript file
                transcript_file = os.path.join('transcripts', f"{os.path.splitext(os.path.basename(self.audio_file))[0]}.txt")
                if os.path.exists(transcript_file):
                    new_transcript_file = os.path.join(os.path.dirname(file_name), f"{os.path.splitext(os.path.basename(file_name))[0]}.txt")
                    shutil.copy2(transcript_file, new_transcript_file)
                    QMessageBox.information(self, 'Export Successful',
                                            f'Audio exported to {file_name}\n'
                                            f'Transcript copied to {new_transcript_file}')
                else:
                    QMessageBox.information(self, 'Export Successful',
                                            f'Audio exported to {file_name}\n'
                                            f'No transcript file found.')
        except Exception as e:
            QMessageBox.warning(self, 'Export Error', str(e))

    def transcribe_audio(self):
        self.transcription_label.setText('Transcribing audio...')
        QApplication.processEvents()

        temp_16k_file = 'temp_16k_audio.wav'
        audio = AudioSegment.from_file(self.audio_file)
        audio_16k = audio.set_frame_rate(16000)
        audio_16k.export(temp_16k_file, format='wav')

        whisper_executable = './whisper.cpp/main'
        model_path = './whisper.cpp/models/ggml-small.bin'

        command = [
            whisper_executable,
            '-m', model_path,
            '-f', temp_16k_file,
            '-otxt'
        ]

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            transcription_file = f'{temp_16k_file}.txt'
            if os.path.exists(transcription_file):
                with open(transcription_file, 'r') as f:
                    transcription = f.read().strip()

                if transcription:
                    self.transcription_label.setText(transcription)
                    self.metadata_button.setEnabled(True)
                    QMessageBox.information(self, 'Transcription Complete', 'Audio transcription completed.')

                    # Generate metadata
                    metadata = self.generate_metadata(transcription)

                    # Save transcription and rename files
                    self.save_transcription_and_rename(transcription, metadata)
                else:
                    error_msg = "Error: Transcription is empty."
                    self.transcription_label.setText(error_msg)
                    QMessageBox.warning(self, 'Error', error_msg)
            else:
                error_msg = f"Error: '{transcription_file}' file not found after transcription."
                self.transcription_label.setText(error_msg)
                QMessageBox.warning(self, 'Error', error_msg)
        except subprocess.CalledProcessError as e:
            error_msg = f'Transcription failed. Error: {str(e)}\nStdout: {e.stdout}\nStderr: {e.stderr}'
            self.transcription_label.setText(error_msg)
            QMessageBox.warning(self, 'Error', error_msg)
        finally:
            # Clean up temporary files
            if os.path.exists(temp_16k_file):
                os.remove(temp_16k_file)
            transcription_file = f'{temp_16k_file}.txt'
            if os.path.exists(transcription_file):
                os.remove(transcription_file)

    def generate_metadata(self, transcription, max_retries=2):
        print(f"Transcription type: {type(transcription)}")
        print(f"Transcription value: {transcription}")

        if not transcription or isinstance(transcription, bool):
            print("Error: Transcription is empty, invalid, or a boolean value")
            return f"{datetime.datetime.now().strftime('%m%d%y')}_no_metadata_1.mp3"

        # Clean the transcription: remove special characters and split into words
        cleaned_words = re.findall(r'\b[a-zA-Z]+\b', str(transcription))
        truncated_text = ' '.join(cleaned_words[:500])

        prompt = f"""Summarize the text using three words and underscore separators only (e.g., hair_metal_pomposity, prince_performs_badly, metallica_on_rise)###text: Prince just won an award. It was pretty incredible###summary: prince_wins_award###text: Metallica is on the rise###summary: metallica_on_rise###text: {truncated_text}###summary:"""

        for attempt in range(max_retries):
            try:
                response = requests.post('http://localhost:8080/completion', json={
                    'prompt': prompt,
                    'n_predict': 10,
                    'temperature': 0.7,
                    'top_k': 40,
                    'top_p': 0.95,
                    'repeat_penalty': 1.1,
                    'stop': ['###'],  # Stop generation at '###'
                    'n_threads': 4  # Reduce number of threads
                })
                
                if response.status_code == 200:
                    result = response.json()
                    summary = result['content'].strip().lower()
                    
                    # Check if the summary matches the expected format
                    if re.match(r'^[a-z]+_[a-z]+_[a-z]+$', summary):
                        date = datetime.datetime.now().strftime('%m%d%y')
                        metadata = f"{date}_{summary}_1.mp3"
                        return metadata
                    else:
                        print(f"Attempt {attempt + 1}: Generated summary '{summary}' does not match expected format. Retrying...")
                else:
                    print(f"Error from llama.cpp server: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Error connecting to llama.cpp server: {e}")
        
        # If both attempts fail, use the first three words of the cleaned transcription
        fallback_words = cleaned_words[:3]
        if len(fallback_words) < 3:
            # If we don't have 3 words, pad with "no", "metadata", "available"
            fallback_words.extend(["no", "metadata", "available"][:3 - len(fallback_words)])
        fallback_summary = '_'.join(word.lower() for word in fallback_words[:3])
        date = datetime.datetime.now().strftime('%m%d%y')
        metadata = f"{date}_{fallback_summary}_1.mp3"
        print(f"Using fallback metadata: {metadata}")
        return metadata

    def save_transcription_and_rename(self, transcription, metadata):
        if not metadata:
            metadata = f"{datetime.datetime.now().strftime('%m%d%y')}_unknown_1.mp3"

        # Create transcripts folder if it doesn't exist
        os.makedirs('transcripts', exist_ok=True)

        # Save transcription
        transcript_file = os.path.join('transcripts', f"{metadata[:-4]}.txt")
        with open(transcript_file, 'w') as f:
            f.write(transcription)

        # Rename audio file
        new_audio_file = metadata
        os.rename(self.audio_file, new_audio_file)
        self.audio_file = new_audio_file

        QMessageBox.information(self, 'Files Saved', f'Transcription saved as {transcript_file}\nAudio file renamed to {new_audio_file}')

    def load_audio(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "", "Audio Files (*.wav *.mp3)", options=options
        )
        if file_name:
            self.audio_file = file_name
            self.play_button.setEnabled(True)
            self.export_button.setEnabled(True)
            self.transcribe_button.setEnabled(True)

            # Load audio data for playback
            self.audio_data, self.fs = sf.read(self.audio_file, dtype='float32')

            # Load associated transcript if it exists
            transcript_file = os.path.join('transcripts', f"{os.path.splitext(os.path.basename(file_name))[0]}.txt")
            if os.path.exists(transcript_file):
                with open(transcript_file, 'r') as f:
                    transcription = f.read()
                self.transcription_label.setText(transcription)
            else:
                self.transcription_label.setText("No transcript available for this audio file.")

    def closeEvent(self, event):
        # Clean up temporary files
        temp_files = ['recorded_audio.wav', 'recorded_audio.txt', 'metadata.txt', 'prompt.txt']
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AudioRecorderApp()
    window.show()
    sys.exit(app.exec_())
