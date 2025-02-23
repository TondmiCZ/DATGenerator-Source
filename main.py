import wave
import os
import numpy as np
import struct
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.signal import savgol_filter
import tkinter.ttk as ttk

__author__ = "TondmiCZ"
__version__ = "1.0"
__copyright__ = "2025 by TondmiCZ"

def process_wav_for_mimic(input_wav, output_dat, failed_files):
    try:
        with wave.open(input_wav, 'rb') as wav_file:
            num_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_rate = wav_file.getframerate()
            num_frames = wav_file.getnframes()
            raw_data = wav_file.readframes(num_frames)
            
            if sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                raise ValueError("Unsupported audio bit depth")
            
            audio_data = np.frombuffer(raw_data, dtype=dtype)
            if num_channels > 1:
                audio_data = audio_data[::num_channels]
            
            max_amplitude = np.max(np.abs(audio_data))
            normalized_amplitude = np.clip(np.abs(audio_data) / (max_amplitude if max_amplitude > 0 else 1) * 255, 0, 255)
            window_size = int(np.clip(np.mean(np.abs(np.diff(normalized_amplitude))) * 150, 25, 101))
            smoothed_amplitude = savgol_filter(normalized_amplitude, window_size, 3)
            smoothed_amplitude = np.clip(smoothed_amplitude, 0, None)
            smoothed_amplitude = np.power(smoothed_amplitude, 1.05)
            frame_duration = frame_rate // 65
            frame_values = np.array_split(smoothed_amplitude, len(audio_data) // frame_duration)
            
            dat_frames = []
            for i, frame in enumerate(frame_values):
                loudness = np.mean(frame)
                loudness_change = np.abs(np.max(frame) - np.min(frame))
                random_movement = np.random.uniform(-2, 2)
                mouth_width = int(np.clip(loudness * 1.3 + loudness_change * 0.7 + random_movement, 0, 255))
                mouth_openness = int(np.clip(loudness * 1.7 + loudness_change * 0.9 + random_movement, 0, 255))
                lip_corners = int(np.clip(127 - (loudness - 127) / 0.5 + random_movement, 0, 255))
                eyebrow = int(np.clip(127 - (loudness - 127) / 30 + np.random.uniform(-1, 1), 121, 130))
                eye_openness = int(np.clip(200 - loudness_change * 1.3 + random_movement, 175, 210))
                if np.random.rand() < 0.02:
                    eye_openness = 150

                eye_lid = 0 if i > 10 else 10

                unknown_value = 257
                dat_frames.append((mouth_width, mouth_openness, lip_corners, eye_lid, eye_openness, eyebrow, unknown_value))

            
            with open(output_dat, "wb") as f:
                f.write(struct.pack("<II", len(dat_frames), 0))
                for frame in dat_frames:
                    f.write(struct.pack("<BBBBBBH", *frame))
    except Exception as e:
        failed_files.append(input_wav)

def select_wav():
    file_paths = filedialog.askopenfilenames(filetypes=[("WAV Files", "*.wav")])
    if file_paths:
        entry_wav.delete(0, tk.END)
        entry_wav.insert(0, ";".join(file_paths))

def select_output():
    output_dir = filedialog.askdirectory()
    if output_dir:
        entry_dat.delete(0, tk.END)
        entry_dat.insert(0, output_dir)

def run_conversion():
    input_wavs = entry_wav.get().strip().split(";")
    output_folder = entry_dat.get().strip()
    
    if not input_wavs or input_wavs == [""]:
        messagebox.showwarning("Error", "Select input file(s).")
        return
    
    if not output_folder:
        messagebox.showwarning("Error", "Select output folder.")
        return
    
    last_output_dat = None
    failed_files = []
    for input_wav in input_wavs:
        last_output_dat = os.path.join(output_folder, os.path.basename(input_wav).rsplit(".", 1)[0] + ".dat")
        process_wav_for_mimic(input_wav, last_output_dat, failed_files)
    
    if last_output_dat:
        messagebox.showinfo("Done", f"Facial expressions generated and saved to {last_output_dat}")
    
    if failed_files:
        messagebox.showwarning("Error", f"The following files could not be processed.:\n" + "\n".join(failed_files))

# GUI aplikace
root = tk.Tk()
root.title("Mafia Facial Expression Maker ")
root.geometry("600x300")
root.resizable(False, False)
frame = ttk.Frame(root, padding=15)
frame.pack(pady=20)

label_wav = ttk.Label(frame, text="Select WAV file:")
label_wav.grid(row=0, column=0, sticky="w")
entry_wav = ttk.Entry(frame, width=60)
entry_wav.grid(row=0, column=1, padx=5)
btn_wav = ttk.Button(frame, text="Browse", command=select_wav)
btn_wav.grid(row=0, column=2)

label_dat = ttk.Label(frame, text="Select output folder:")
label_dat.grid(row=1, column=0, sticky="w")
entry_dat = ttk.Entry(frame, width=60)
entry_dat.grid(row=1, column=1, padx=5)
btn_dat = ttk.Button(frame, text="Browse", command=select_output)
btn_dat.grid(row=1, column=2)

btn_convert = ttk.Button(root, text="Generate Facial Expressions", command=run_conversion)
btn_convert.pack(pady=10)

status_label = ttk.Label(root, text="© 2025 by TondmiCZ", relief=tk.SUNKEN, anchor="w")
status_label.pack(fill=tk.X, side=tk.BOTTOM)

root.mainloop()
