import wave
import os
import numpy as np
import struct
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from scipy.signal import savgol_filter, spectrogram, find_peaks
from scipy.ndimage import gaussian_filter1d
import threading
import json

__author__ = "TondmiCZ"
__version__ = "2.0"
__copyright__ = "2025 by TondmiCZ"

class FacialExpressionMaker:
    def __init__(self, root):
        self.root = root
        self.setup_dark_theme()
        self.setup_gui()
        self.processing = False
        
    def setup_dark_theme(self):
        """Nastavení tmavého režimu"""
        self.root.configure(bg='#2b2b2b')
        
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='#ffffff')
        style.configure('TButton', background='#404040', foreground='#ffffff')
        style.map('TButton', background=[('active', '#505050')])
        
        
        style.configure('TEntry', background='#ffffff', foreground='#000000', insertcolor='#000000')
        
        style.configure('TProgressbar', background='#00aa00', troughcolor='#404040')
        
    def setup_gui(self):
        self.root.title("Mafia Facial Expression Maker v2.0")
        self.root.geometry("700x300")
        self.root.resizable(False, False)
        
        frame = ttk.Frame(self.root, padding=15)
        frame.pack(pady=20)
        ttk.Label(frame, text="WAV soubory:").grid(row=0, column=0, sticky="w")
        self.entry_wav = ttk.Entry(frame, width=60)
        self.entry_wav.grid(row=0, column=1, padx=5)
        ttk.Button(frame, text="Procházet", command=self.select_wav).grid(row=0, column=2)

        ttk.Label(frame, text="Výstupní složka:").grid(row=1, column=0, sticky="w")
        self.entry_output = ttk.Entry(frame, width=60)
        self.entry_output.grid(row=1, column=1, padx=5)
        ttk.Button(frame, text="Procházet", command=self.select_output).grid(row=1, column=2)


        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", padx=20, pady=10)

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        self.btn_convert = ttk.Button(btn_frame, text="Generovat mimiku", command=self.start_conversion)
        self.btn_convert.pack(side="left", padx=5)
        
        self.btn_cancel = ttk.Button(btn_frame, text="Zrušit", command=self.cancel_conversion, state="disabled")
        self.btn_cancel.pack(side="left", padx=5)

      
        self.status_var = tk.StringVar(value="Připraven")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
    def select_wav(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Audio soubory", "*.wav")])
        if file_paths:
            self.entry_wav.delete(0, tk.END)
            self.entry_wav.insert(0, ";".join(file_paths))
    
    def select_output(self):
        output_dir = filedialog.askdirectory()
        if output_dir:
            self.entry_output.delete(0, tk.END)
            self.entry_output.insert(0, output_dir)
    
    def detect_speech_segments(self, audio_data, sample_rate):
        """Detekuje segmenty s řečí vs ticho"""
        threshold = 0.02 * np.max(np.abs(audio_data))
        
        window_size = int(sample_rate * 0.02)  
        rms_values = []
        
        for i in range(0, len(audio_data) - window_size, window_size // 2):
            window = audio_data[i:i + window_size]
            
            if len(window) > 0:
                mean_square = np.mean(window.astype(np.float64) ** 2)
                if mean_square >= 0 and not np.isnan(mean_square):
                    rms = np.sqrt(mean_square)
                else:
                    rms = 0.0
            else:
                rms = 0.0
                
            rms_values.append(rms)
        
        speech_mask = np.array(rms_values) > threshold
        return speech_mask, rms_values
    
    def analyze_frequencies(self, audio_data, sample_rate):
        """Frekvenční analýza pro detekci různých hlásek"""
        f, t, Sxx = spectrogram(audio_data, sample_rate, nperseg=1024)
        
        low_freq_mask = (f >= 100) & (f <= 500)
        mid_freq_mask = (f >= 500) & (f <= 2000)
        high_freq_mask = (f >= 2000) & (f <= 8000)
        
        low_energy = np.mean(Sxx[low_freq_mask, :], axis=0)
        mid_energy = np.mean(Sxx[mid_freq_mask, :], axis=0)
        high_energy = np.mean(Sxx[high_freq_mask, :], axis=0)
        
        return t, low_energy, mid_energy, high_energy
    
    def generate_advanced_facial_data(self, audio_data, sample_rate):
        
        # Detekce řeč
        speech_mask, rms_values = self.detect_speech_segments(audio_data, sample_rate)
        
        # Frekvenční analýza
        spec_time, low_energy, mid_energy, high_energy = self.analyze_frequencies(audio_data, sample_rate)
        
        # Normalizace amplitudy
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude == 0:
            return []
        
        normalized_amplitude = np.abs(audio_data) / max_amplitude * 255
        
        # Vyhlazení
        window_size = int(np.clip(np.mean(np.abs(np.diff(normalized_amplitude))) * 100, 15, 51))
        if window_size % 2 == 0:
            window_size += 1
            
        smoothed_amplitude = savgol_filter(normalized_amplitude, window_size, 3)
        smoothed_amplitude = gaussian_filter1d(smoothed_amplitude, sigma=2.0)
        smoothed_amplitude = np.clip(smoothed_amplitude, 0, None)
        
        # Detekce píků pro mrkaní
        peaks, _ = find_peaks(smoothed_amplitude, height=np.mean(smoothed_amplitude) * 1.2, distance=sample_rate//15)
        
        # Rozdělení do snímků
        frame_duration = sample_rate // 65
        frame_values = np.array_split(smoothed_amplitude, len(audio_data) // frame_duration)
        total_frames = len(frame_values)
        
        # Přirozené cykly
        breathing_cycle = np.sin(np.linspace(0, total_frames / 35 * 2 * np.pi, total_frames)) * 6
        subtle_movement = np.sin(np.linspace(0, total_frames / 80 * 2 * np.pi, total_frames)) * 3
        
        dat_frames = []
        
        # Stavové proměnné
        mouth_state = {
            'last_width': 30,
            'last_openness': 20,
            'speech_intensity': 0,
            'rest_timer': 0,
            'expression_type': 'neutral'
        }
        
        blink_cooldown = 0
        last_volume_spike = 0
        
        # Analýza celkového charakteru audio
        overall_loudness = np.mean(smoothed_amplitude)
        
        for i, frame in enumerate(frame_values):
            if len(frame) == 0:
                continue
            
            # Generování různých gest
            gesture_intensity = np.sin(i * 0.3) * 8
            quick_gesture = np.sin(i * 1.2) * 5
            speech_rhythm = np.sin(i * 0.8) * 6
            
            # Analýza současného snímku
            current_loudness = np.mean(frame)
            loudness_change = np.abs(np.max(frame) - np.min(frame)) if len(frame) > 1 else 0
            volume_spike = current_loudness > overall_loudness * 1.4
            
            # Frekvenční data
            current_time = (i * frame_duration) / sample_rate
            if len(spec_time) > 1:
                low_interp = np.interp(current_time, spec_time, low_energy)
                mid_interp = np.interp(current_time, spec_time, mid_energy)
                high_interp = np.interp(current_time, spec_time, high_energy)
            else:
                low_interp = mid_interp = high_interp = 0
            
            # Detekce typu řeči
            base_threshold = 0.02 * 100
            is_speaking = current_loudness > base_threshold
            is_whisper = current_loudness > base_threshold * 0.3 and current_loudness < base_threshold
            is_loud_speech = current_loudness > overall_loudness * 1.1
            
            # Backup detekce
            has_any_sound = (current_loudness > 0.02 * 20 or 
                           loudness_change > 2 or 
                           low_interp > 0.1 or 
                           mid_interp > 0.1 or 
                           high_interp > 0.1)
            
            if has_any_sound and not is_speaking and not is_whisper:
                is_whisper = True
                
            if mouth_state['speech_intensity'] > 0.2 and current_loudness > base_threshold * 0.1:
                is_whisper = True
            
            # Logika pro ústa
            if is_speaking or is_whisper:
                mouth_state['rest_timer'] = 0
                mouth_state['speech_intensity'] = min(mouth_state['speech_intensity'] + 0.2, 1.0)
                
                if is_loud_speech:
                    base_width = np.clip(80 + current_loudness * 1.3 + low_interp * 0.4 + gesture_intensity, 70, 180)
                    base_openness = np.clip(60 + current_loudness * 1.7 + mid_interp * 0.35 + quick_gesture, 50, 140)
                    mouth_state['expression_type'] = 'loud'
                    
                elif is_whisper:
                    base_width = np.clip(40 + current_loudness * 1.0 + gesture_intensity * 0.5, 35, 100)
                    base_openness = np.clip(30 + current_loudness * 1.3 + quick_gesture * 0.3, 25, 80)
                    mouth_state['expression_type'] = 'whisper'
                    
                else:
                    base_width = np.clip(60 + current_loudness * 1.3 + low_interp * 0.3 + gesture_intensity, 45, 160)
                    base_openness = np.clip(50 + current_loudness * 1.7 + loudness_change * 0.5 + speech_rhythm, 35, 130)
                    mouth_state['expression_type'] = 'normal'
                
                # Modulace podle typu hlásek
                if high_interp > mid_interp * 1.2:
                    base_width *= 0.8
                    base_openness *= 0.7
                    base_width += quick_gesture * 0.8
                    
                elif low_interp > mid_interp * 1.3:
                    base_width *= 1.15
                    base_openness *= 1.25
                    base_openness += gesture_intensity * 0.6
                
                # Extra gestikulace
                if i % 25 == 0:
                    gesture_type = np.random.choice(['smile', 'emphasis', 'pause'])
                    if gesture_type == 'smile':
                        base_width += np.random.uniform(5, 15)
                    elif gesture_type == 'emphasis':
                        base_openness += np.random.uniform(8, 20)
                    elif gesture_type == 'pause':
                        base_width *= 0.85
                        base_openness *= 0.8
                        
            else:
                # Při tichu
                mouth_state['rest_timer'] += 1
                mouth_state['speech_intensity'] = max(mouth_state['speech_intensity'] - 0.05, 0)
                
                residual_activity = current_loudness > 0.02 * 10
                
                if mouth_state['rest_timer'] < 50:
                    transition_factor = mouth_state['rest_timer'] / 50
                    base_width = mouth_state['last_width'] * (1 - transition_factor * 0.4)
                    base_openness = mouth_state['last_openness'] * (1 - transition_factor * 0.5)
                    
                elif residual_activity:
                    base_width = 35 + current_loudness * 0.3 + gesture_intensity * 0.3
                    base_openness = 20 + current_loudness * 0.2 + quick_gesture * 0.2
                    
                elif mouth_state['rest_timer'] > 120 and np.random.rand() < 0.06:
                    gesture_strength = np.random.uniform(0.5, 1.5)
                    base_width = 50 + np.random.uniform(10, 30) + gesture_intensity * gesture_strength
                    base_openness = 35 + np.random.uniform(10, 25) + quick_gesture * gesture_strength
                    mouth_state['expression_type'] = 'breathing'
                    mouth_state['rest_timer'] = 0
                    
                else:
                    base_width = 25 + breathing_cycle[i] * 0.4 + subtle_movement[i] * 0.6 + gesture_intensity * 0.2
                    base_openness = 12 + breathing_cycle[i] * 0.3 + subtle_movement[i] * 0.4 + quick_gesture * 0.1
                    mouth_state['expression_type'] = 'rest'
            
            # Vyhlazení a omezení
            micro_variation = np.random.uniform(-1.5, 1.5)
            momentum = 0.45 if (is_speaking or is_whisper or has_any_sound) else 0.6
            target_width = base_width + micro_variation
            target_openness = base_openness + micro_variation
            
            max_change_width = 35 if (is_speaking or is_whisper or has_any_sound) else 8
            max_change_openness = 30 if (is_speaking or is_whisper or has_any_sound) else 6
            
            width_change = target_width - mouth_state['last_width']
            openness_change = target_openness - mouth_state['last_openness']
            
            if abs(width_change) > max_change_width:
                width_change = np.sign(width_change) * max_change_width
            if abs(openness_change) > max_change_openness:
                openness_change = np.sign(openness_change) * max_change_openness
            
            mouth_width = mouth_state['last_width'] + width_change * momentum
            mouth_openness = mouth_state['last_openness'] + openness_change * momentum
            
            mouth_width = int(np.clip(mouth_width, 15, 190))
            mouth_openness = int(np.clip(mouth_openness, 8, 150))
            
            if volume_spike and i - last_volume_spike > 10:
                mouth_openness = min(mouth_openness + 35, 150)
                mouth_width = min(mouth_width + 25, 190)
                last_volume_spike = i
            
            # Rty a výrazy
            base_lip = 127 + subtle_movement[i] * 0.4
            
            if is_speaking or is_whisper:
                if mouth_state['expression_type'] == 'loud':
                    lip_corners = int(np.clip(base_lip - 8 + high_interp * 0.1, 105, 150))
                elif mouth_state['expression_type'] == 'whisper':
                    lip_corners = int(np.clip(base_lip + 5, 120, 140))
                else:
                    lip_corners = int(np.clip(base_lip + (current_loudness - 127) * 0.1 + high_interp * 0.08, 110, 145))
            else:
                lip_corners = int(np.clip(base_lip + micro_variation * 0.5, 120, 135))
            
            # Obočí
            base_brow = 127 + breathing_cycle[i] * 0.15
            
            if is_loud_speech:
                eyebrow = int(np.clip(base_brow - 4 + np.random.uniform(-1, 1), 122, 132))
            else:
                eyebrow = int(np.clip(base_brow + micro_variation * 0.3, 125, 129))
            
            # Oči a mrkaní
            base_eye = 195 + breathing_cycle[i] * 0.6
            
            if blink_cooldown > 0:
                blink_cooldown -= 1
            
            should_blink = False
            
            if i in peaks and blink_cooldown == 0:
                should_blink = True
                blink_cooldown = 20
            elif mouth_state['rest_timer'] > 100 and np.random.rand() < 0.008 and blink_cooldown == 0:
                should_blink = True
                blink_cooldown = np.random.randint(60, 150)
            elif (is_speaking or is_whisper) and np.random.rand() < 0.004 and blink_cooldown == 0:
                should_blink = True
                blink_cooldown = np.random.randint(40, 100)
            
            if should_blink:
                eye_openness = np.random.choice([145, 160, 155])
            elif (is_speaking or is_whisper) and loudness_change > 80:
                eye_adjustment = loudness_change * 1.3 * 0.4
                eye_openness = int(np.clip(base_eye - eye_adjustment + micro_variation, 180, 210))
            else:
                eye_openness = int(np.clip(base_eye + micro_variation, 180, 210))
            
            # Víčka
            eye_lid = 0 if i > 10 else 10
            
            # Uložení stavu
            mouth_state['last_width'] = mouth_width
            mouth_state['last_openness'] = mouth_openness
            
            # Sestavení snímku
            unknown_value = 257
            dat_frames.append((mouth_width, mouth_openness, lip_corners, eye_lid, eye_openness, eyebrow, unknown_value))
        
        return dat_frames
    
    def process_wav_for_mimic(self, input_wav, output_dat, progress_callback=None):
        """Zpracování WAV souboru"""
        try:
            self.status_var.set(f"Načítám {os.path.basename(input_wav)}...")
            
            with wave.open(input_wav, 'rb') as wav_file:
                num_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                num_frames = wav_file.getnframes()
                raw_data = wav_file.readframes(num_frames)
                
                if progress_callback:
                    progress_callback(15)
                
                if sample_width == 2:
                    dtype = np.int16
                elif sample_width == 4:
                    dtype = np.int32
                else:
                    raise ValueError("Nepodporovaná bitová hloubka")
                
                audio_data = np.frombuffer(raw_data, dtype=dtype)
                if num_channels > 1:
                    audio_data = audio_data[::num_channels]
                
                if progress_callback:
                    progress_callback(25)
                
                self.status_var.set(f"Analyzuji {os.path.basename(input_wav)}...")
                
                # Pokročilá analýza
                dat_frames = self.generate_advanced_facial_data(audio_data, frame_rate)
                
                if progress_callback:
                    progress_callback(85)
                
                self.status_var.set(f"Ukládám {os.path.basename(input_wav)}...")
                
                with open(output_dat, "wb") as f:
                    f.write(struct.pack("<II", len(dat_frames), 0))
                    for frame in dat_frames:
                        f.write(struct.pack("<BBBBBBH", *frame))
                
                if progress_callback:
                    progress_callback(100)
                    
                return True
                
        except Exception as e:
            self.status_var.set(f"Chyba: {str(e)}")
            return False
    
    def start_conversion(self):
        """Spustí konverzi"""
        if self.processing:
            return
            
        input_files = [f.strip() for f in self.entry_wav.get().split(";") if f.strip()]
        output_folder = self.entry_output.get().strip()
        
        if not input_files:
            messagebox.showwarning("Chyba", "Vyberte vstupní soubory.")
            return
        
        if not output_folder:
            messagebox.showwarning("Chyba", "Vyberte výstupní složku.")
            return
        
        self.processing = True
        self.btn_convert.config(state="disabled")
        self.btn_cancel.config(state="normal")
        
        thread = threading.Thread(target=self.conversion_worker, args=(input_files, output_folder))
        thread.daemon = True
        thread.start()
    
    def conversion_worker(self, input_files, output_folder):
        """Worker pro konverzi souborů"""
        failed_files = []
        total_files = len(input_files)
        
        for i, input_file in enumerate(input_files):
            if not self.processing:
                break
                
            output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0] + ".dat")
            
            def progress_update(progress):
                total_progress = (i / total_files) * 100 + (progress / total_files)
                self.progress_var.set(total_progress)
            
            success = self.process_wav_for_mimic(input_file, output_file, progress_update)
            if not success:
                failed_files.append(input_file)
        
        self.processing = False
        self.btn_convert.config(state="normal")
        self.btn_cancel.config(state="disabled")
        self.progress_var.set(0)
        
        if failed_files:
            messagebox.showwarning("Varování", f"Tyto soubory se nepodařilo zpracovat:\n" + "\n".join(failed_files))
        else:
            messagebox.showinfo("Hotovo", f"Úspěšně zpracováno {total_files} souborů!")
                
        self.status_var.set("Připraven")
    
    def cancel_conversion(self):
        """Zruší probíhající konverzi"""
        self.processing = False
        self.status_var.set("Zrušeno...")

def main():
    root = tk.Tk()
    app = FacialExpressionMaker(root)
    root.mainloop()

if __name__ == "__main__":
    main()
