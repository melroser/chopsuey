"""
chopsuey_gui.py - Interactive GUI for the Chopsuey vocal chop extractor

Provides a visual interface for:
- Real-time visualization during processing
- Interactive browsing of generated chops
- Waveform visualization and playback
- Selection and export of chops
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
from pathlib import Path
import json
import numpy as np
import librosa
import sounddevice as sd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from datetime import datetime
import os
import shutil

# Import the existing chopsuey modules
from chopsuey import VocalChopExtractor
from chopsuey_visualizer import ChopsueyVisualizer


class ChopsueyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chopsuey - Visual Vocal Chop Extractor")
        self.root.geometry("1400x900")
        
        # Dark theme colors matching the visualizer
        self.colors = {
            'bg': '#1e1e1e',
            'fg': '#ffffff',
            'accent': '#00ffff',
            'secondary': '#ff6b6b',
            'tertiary': '#4ecdc4',
            'panel': '#2d2d2d',
            'button': '#3d3d3d',
            'button_hover': '#4d4d4d',
            'selected': '#5d5d5d'
        }
        
        # Apply dark theme
        self.root.configure(bg=self.colors['bg'])
        
        # Processing state
        self.processing = False
        self.current_audio = None
        self.current_sr = None
        self.selected_chops = []
        self.all_chops = {}
        self.process_queue = queue.Queue()
        
        # Initialize extractor and visualizer
        self.extractor = None
        self.output_dir = "output_gui"
        
        # Setup GUI
        self.setup_gui()
        self.setup_styles()
        
        # Start queue processor
        self.root.after(100, self.process_queue_items)
        
    def setup_styles(self):
        """Configure ttk styles for dark theme"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors for various widgets
        style.configure('Dark.TFrame', background=self.colors['bg'])
        style.configure('Panel.TFrame', background=self.colors['panel'])
        style.configure('Dark.TLabel', background=self.colors['bg'], 
                       foreground=self.colors['fg'])
        style.configure('Panel.TLabel', background=self.colors['panel'], 
                       foreground=self.colors['fg'])
        style.configure('Accent.TLabel', background=self.colors['bg'], 
                       foreground=self.colors['accent'], font=('Arial', 12, 'bold'))
        style.configure('Dark.TButton', background=self.colors['button'], 
                       foreground=self.colors['fg'])
        style.map('Dark.TButton',
                  background=[('active', self.colors['button_hover'])])
        
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Main container
        main_container = ttk.Frame(self.root, style='Dark.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top control panel
        self.setup_control_panel(main_container)
        
        # Main content area with paned window
        paned = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left panel - Processing visualization
        left_frame = ttk.Frame(paned, style='Panel.TFrame')
        paned.add(left_frame, weight=2)
        self.setup_processing_panel(left_frame)
        
        # Right panel - Chop browser
        right_frame = ttk.Frame(paned, style='Panel.TFrame')
        paned.add(right_frame, weight=3)
        self.setup_browser_panel(right_frame)
        
        # Bottom status bar
        self.setup_status_bar(main_container)
        
    def setup_control_panel(self, parent):
        """Setup top control panel"""
        control_frame = ttk.Frame(parent, style='Panel.TFrame')
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title
        title = ttk.Label(control_frame, text="ðµ CHOPSUEY", style='Accent.TLabel')
        title.pack(side=tk.LEFT, padx=10)
        
        # File selection
        self.file_var = tk.StringVar(value="No file selected")
        file_label = ttk.Label(control_frame, textvariable=self.file_var, 
                              style='Panel.TLabel')
        file_label.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(control_frame, text="Select ZIP", 
                  command=self.select_file, style='Dark.TButton').pack(side=tk.LEFT, padx=5)
        
        # Processing controls
        self.process_btn = ttk.Button(control_frame, text="Process", 
                                     command=self.start_processing, 
                                     style='Dark.TButton', state='disabled')
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        # Mode selection
        ttk.Label(control_frame, text="Mode:", style='Panel.TLabel').pack(side=tk.LEFT, padx=(20, 5))
        self.mode_var = tk.StringVar(value="hard")
        mode_combo = ttk.Combobox(control_frame, textvariable=self.mode_var, 
                                 values=["normal", "hard", "harder"], 
                                 width=10, state='readonly')
        mode_combo.pack(side=tk.LEFT)
        
        # Export button
        self.export_btn = ttk.Button(control_frame, text="Export Selected", 
                                    command=self.export_selected, 
                                    style='Dark.TButton', state='disabled')
        self.export_btn.pack(side=tk.RIGHT, padx=10)
        
    def setup_processing_panel(self, parent):
        """Setup left panel for processing visualization"""
        # Title
        title_frame = ttk.Frame(parent, style='Panel.TFrame')
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(title_frame, text="Processing Visualization", 
                 style='Accent.TLabel').pack(side=tk.LEFT)
        
        # Progress
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(parent, variable=self.progress_var, 
                                       maximum=100, length=300)
        self.progress.pack(padx=10, pady=5)
        
        # Current file label
        self.current_file_var = tk.StringVar(value="Ready to process...")
        ttk.Label(parent, textvariable=self.current_file_var, 
                 style='Panel.TLabel').pack(pady=5)
        
        # Matplotlib figure for real-time visualization
        self.fig_process = Figure(figsize=(6, 8), facecolor=self.colors['panel'])
        self.canvas_process = FigureCanvasTkAgg(self.fig_process, parent)
        self.canvas_process.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize empty plots
        self.setup_process_plots()
        
    def setup_process_plots(self):
        """Setup the processing visualization plots"""
        self.fig_process.clear()
        
        # Create subplots
        gs = self.fig_process.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
        
        # Waveform plot
        self.ax_wave = self.fig_process.add_subplot(gs[0])
        self.ax_wave.set_title("Waveform", color=self.colors['fg'])
        self.ax_wave.set_facecolor(self.colors['bg'])
        self.ax_wave.tick_params(colors=self.colors['fg'])
        
        # Spectrogram plot
        self.ax_spec = self.fig_process.add_subplot(gs[1])
        self.ax_spec.set_title("Spectrogram", color=self.colors['fg'])
        self.ax_spec.set_facecolor(self.colors['bg'])
        self.ax_spec.tick_params(colors=self.colors['fg'])
        
        # Chop detection plot
        self.ax_chops = self.fig_process.add_subplot(gs[2])
        self.ax_chops.set_title("Detected Chops", color=self.colors['fg'])
        self.ax_chops.set_facecolor(self.colors['bg'])
        self.ax_chops.tick_params(colors=self.colors['fg'])
        
        self.canvas_process.draw()
        
    def setup_browser_panel(self, parent):
        """Setup right panel for browsing chops"""
        # Title and controls
        title_frame = ttk.Frame(parent, style='Panel.TFrame')
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(title_frame, text="Chop Browser", 
                 style='Accent.TLabel').pack(side=tk.LEFT)
        
        # Filter controls
        filter_frame = ttk.Frame(parent, style='Panel.TFrame')
        filter_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(filter_frame, text="Filter:", style='Panel.TLabel').pack(side=tk.LEFT, padx=5)
        self.filter_var = tk.StringVar(value="all")
        filter_combo = ttk.Combobox(filter_frame, textvariable=self.filter_var, 
                                   values=["all", "oneshot", "vocal_phrase", "loop", 
                                          "drop", "atmosphere", "ear_candy"], 
                                   width=15, state='readonly')
        filter_combo.pack(side=tk.LEFT, padx=5)
        filter_combo.bind('<<ComboboxSelected>>', self.filter_chops)
        
        # Chop list with scrollbar
        list_frame = ttk.Frame(parent, style='Panel.TFrame')
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create treeview for chop list
        columns = ('Type', 'Duration', 'Text', 'Features')
        self.chop_tree = ttk.Treeview(list_frame, columns=columns, 
                                     show='tree headings', height=10)
        
        # Configure columns
        self.chop_tree.heading('#0', text='ID')
        self.chop_tree.heading('Type', text='Type')
        self.chop_tree.heading('Duration', text='Duration')
        self.chop_tree.heading('Text', text='Text')
        self.chop_tree.heading('Features', text='Key Features')
        
        self.chop_tree.column('#0', width=80)
        self.chop_tree.column('Type', width=100)
        self.chop_tree.column('Duration', width=80)
        self.chop_tree.column('Text', width=200)
        self.chop_tree.column('Features', width=150)
        
        # Scrollbars
        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=self.chop_tree.yview)
        hsb = ttk.Scrollbar(list_frame, orient="horizontal", command=self.chop_tree.xview)
        self.chop_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Pack
        self.chop_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        
        # Bind selection event
        self.chop_tree.bind('<<TreeviewSelect>>', self.on_chop_select)
        
        # Waveform viewer
        wave_frame = ttk.Frame(parent, style='Panel.TFrame')
        wave_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Matplotlib figure for selected chop
        self.fig_chop = Figure(figsize=(6, 3), facecolor=self.colors['panel'])
        self.canvas_chop = FigureCanvasTkAgg(self.fig_chop, wave_frame)
        self.canvas_chop.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Playback controls
        control_frame = ttk.Frame(parent, style='Panel.TFrame')
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.play_btn = ttk.Button(control_frame, text="â¶ Play", 
                                  command=self.play_chop, 
                                  style='Dark.TButton', state='disabled')
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="â  Stop", 
                                  command=self.stop_playback, 
                                  style='Dark.TButton', state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.select_btn = ttk.Button(control_frame, text="+ Add to Selection", 
                                    command=self.add_to_selection, 
                                    style='Dark.TButton', state='disabled')
        self.select_btn.pack(side=tk.LEFT, padx=20)
        
        # Selection count
        self.selection_var = tk.StringVar(value="0 chops selected")
        ttk.Label(control_frame, textvariable=self.selection_var, 
                 style='Panel.TLabel').pack(side=tk.RIGHT, padx=10)
        
    def setup_status_bar(self, parent):
        """Setup bottom status bar"""
        status_frame = ttk.Frame(parent, style='Panel.TFrame')
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                style='Panel.TLabel')
        status_label.pack(side=tk.LEFT, padx=10)
        
    def select_file(self):
        """Open file dialog to select ZIP file"""
        filename = filedialog.askopenfilename(
            title="Select ZIP file containing songs",
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")]
        )
        
        if filename:
            self.file_var.set(Path(filename).name)
            self.input_file = filename
            self.process_btn['state'] = 'normal'
            self.status_var.set(f"Selected: {Path(filename).name}")
            
    def start_processing(self):
        """Start processing in a separate thread"""
        if self.processing:
            return
            
        self.processing = True
        self.process_btn['state'] = 'disabled'
        self.all_chops.clear()
        self.chop_tree.delete(*self.chop_tree.get_children())
        
        # Start processing thread
        thread = threading.Thread(target=self.process_files)
        thread.daemon = True
        thread.start()
        
    def process_files(self):
        """Process files in background thread"""
        try:
            # Initialize extractor with custom visualizer
            self.extractor = VocalChopExtractor(self.output_dir)
            
            # Override the visualizer to send updates to GUI
            self.extractor.visualizer = GUIVisualizer(self.process_queue)
            
            # Import necessary modules for processing
            import zipfile
            import shutil
            from pathlib import Path
            
            temp_dir = Path('temp_extract_gui')
            temp_dir.mkdir(exist_ok=True)
            
            # Extract files
            self.process_queue.put(('status', 'Extracting files...'))
            with zipfile.ZipFile(self.input_file, 'r') as zip_ref:
                members_to_extract = [
                    member for member in zip_ref.infolist()
                    if not member.filename.startswith('__MACOSX/')
                    and '.DS_Store' not in member.filename
                ]
                zip_ref.extractall(temp_dir, members=members_to_extract)
            
            # Find audio files
            audio_files = list(temp_dir.rglob('*.mp3')) + \
                         list(temp_dir.rglob('*.wav')) + \
                         list(temp_dir.rglob('*.flac'))
            
            total_files = len(audio_files)
            
            # Process each file
            for idx, audio_file in enumerate(audio_files):
                progress = (idx / total_files) * 100
                self.process_queue.put(('progress', progress))
                self.process_queue.put(('current_file', audio_file.name))
                
                # Process file
                self.process_single_file(audio_file, idx)
                
            # Cleanup
            shutil.rmtree(temp_dir)
            
            # Complete
            self.process_queue.put(('progress', 100))
            self.process_queue.put(('status', 'Processing complete!'))
            self.process_queue.put(('complete', True))
            
        except Exception as e:
            self.process_queue.put(('error', str(e)))
            
    def process_single_file(self, audio_file, file_idx):
        """Process a single audio file"""
        # Get metadata
        metadata = self.extractor.identify_song(audio_file)
        self.process_queue.put(('status', f"Processing: {metadata['artist']} - {metadata['title']}"))
        
        # Separate vocals
        self.process_queue.put(('stage', 'Separating vocals...'))
        vocals_path = self.extractor.separate_vocals(audio_file)
        
        # Load vocals for visualization
        y, sr = librosa.load(vocals_path, sr=None)
        self.process_queue.put(('audio_data', (y, sr, 'vocals')))
        
        # Transcribe
        self.process_queue.put(('stage', 'Transcribing lyrics...'))
        transcription = self.extractor.transcribe_lyrics(vocals_path)
        
        # Analyze for chops based on mode
        self.process_queue.put(('stage', f'Analyzing for chops ({self.mode_var.get()} mode)...'))
        
        if self.mode_var.get() == "normal":
            chops = self.extractor.analyze_for_chops(vocals_path, transcription, metadata)
        elif self.mode_var.get() == "hard":
            chops = self.extractor.analyze_for_chops_hard(vocals_path, transcription, metadata)
        else:  # harder
            chops = self.extractor.analyze_for_chops_harder(vocals_path, transcription, metadata)
        
        # Send chops visualization
        self.process_queue.put(('chops_detected', chops))
        
        # Save chops
        folder = self.extractor.save_chops(chops, metadata, audio_file.name, 
                                          out_dir=Path(self.output_dir) / self.mode_var.get())
        
        # Store chops for browsing
        for i, chop in enumerate(chops):
            chop_id = f"{file_idx:02d}_{i:03d}"
            self.all_chops[chop_id] = {
                'metadata': metadata,
                'chop': chop,
                'folder': folder,
                'file_idx': file_idx,
                'chop_idx': i
            }
            self.process_queue.put(('add_chop', (chop_id, chop)))
            
    def process_queue_items(self):
        """Process items from the queue"""
        try:
            while True:
                item = self.process_queue.get_nowait()
                command, data = item
                
                if command == 'status':
                    self.status_var.set(data)
                elif command == 'progress':
                    self.progress_var.set(data)
                elif command == 'current_file':
                    self.current_file_var.set(f"Processing: {data}")
                elif command == 'stage':
                    self.current_file_var.set(data)
                elif command == 'audio_data':
                    self.visualize_processing(*data)
                elif command == 'chops_detected':
                    self.visualize_chops(data)
                elif command == 'add_chop':
                    self.add_chop_to_browser(*data)
                elif command == 'complete':
                    self.processing_complete()
                elif command == 'error':
                    messagebox.showerror("Processing Error", data)
                    self.processing_complete()
                    
        except queue.Empty:
            pass
            
        # Schedule next check
        self.root.after(100, self.process_queue_items)
        
    def visualize_processing(self, audio, sr, label):
        """Update processing visualization"""
        # Clear previous plots
        self.ax_wave.clear()
        self.ax_spec.clear()
        
        # Waveform
        time = np.linspace(0, len(audio)/sr, len(audio))
        self.ax_wave.plot(time, audio, color=self.colors['accent'], linewidth=0.5)
        self.ax_wave.set_title(f"Waveform - {label}", color=self.colors['fg'])
        self.ax_wave.set_xlabel("Time (s)", color=self.colors['fg'])
        self.ax_wave.set_ylabel("Amplitude", color=self.colors['fg'])
        self.ax_wave.set_facecolor(self.colors['bg'])
        self.ax_wave.tick_params(colors=self.colors['fg'])
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, 
                                ax=self.ax_spec, cmap='magma')
        self.ax_spec.set_title("Spectrogram", color=self.colors['fg'])
        self.ax_spec.set_facecolor(self.colors['bg'])
        self.ax_spec.tick_params(colors=self.colors['fg'])
        
        self.canvas_process.draw()
        
    def visualize_chops(self, chops):
        """Visualize detected chops"""
        self.ax_chops.clear()
        
        # Create timeline visualization
        colors = {
            'oneshot': '#ff6b6b',
            'vocal_phrase': '#4ecdc4',
            'loop': '#45b7d1',
            'drop': '#f9ca24',
            'atmosphere': '#6c5ce7',
            'ear_candy': '#a29bfe',
            'misc': '#95afc0'
        }
        
        y_pos = 0
        for i, chop in enumerate(chops):
            color = colors.get(chop['type'], '#95afc0')
            self.ax_chops.barh(y_pos, chop['end'] - chop['start'], 
                              left=chop['start'], height=0.8, 
                              color=color, alpha=0.8, 
                              label=chop['type'] if i == 0 else "")
            y_pos += 1
            
        self.ax_chops.set_ylim(-0.5, len(chops) - 0.5)
        self.ax_chops.set_xlabel("Time (s)", color=self.colors['fg'])
        self.ax_chops.set_title(f"Detected Chops ({len(chops)} total)", color=self.colors['fg'])
        self.ax_chops.set_facecolor(self.colors['bg'])
        self.ax_chops.tick_params(colors=self.colors['fg'])
        
        # Add legend
        handles, labels = self.ax_chops.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax_chops.legend(by_label.values(), by_label.keys(), 
                            loc='upper right', framealpha=0.8)
        
        self.canvas_process.draw()
        
    def add_chop_to_browser(self, chop_id, chop):
        """Add a chop to the browser list"""
        # Format duration
        duration = f"{chop['end'] - chop['start']:.2f}s"
        
        # Truncate text
        text = chop['text'][:50] + "..." if len(chop['text']) > 50 else chop['text']
        
        # Key features
        if 'features' in chop:
            features = f"E:{chop['features'].get('mean_energy', 0):.2f} " \
                      f"P:{chop['features'].get('pitch_content', 0):.2f}"
        else:
            features = "N/A"
            
        # Add to tree
        self.chop_tree.insert('', 'end', iid=chop_id, text=chop_id,
                             values=(chop['type'], duration, text, features))
                             
    def filter_chops(self, event=None):
        """Filter displayed chops by type"""
        # Clear current display
        self.chop_tree.delete(*self.chop_tree.get_children())
        
        # Re-add filtered chops
        filter_type = self.filter_var.get()
        for chop_id, data in self.all_chops.items():
            chop = data['chop']
            if filter_type == "all" or chop['type'] == filter_type:
                self.add_chop_to_browser(chop_id, chop)
                
    def on_chop_select(self, event):
        """Handle chop selection"""
        selection = self.chop_tree.selection()
        if not selection:
            return
            
        chop_id = selection[0]
        if chop_id not in self.all_chops:
            return
            
        # Get chop data
        data = self.all_chops[chop_id]
        chop = data['chop']
        
        # Store current audio
        self.current_audio = chop['audio_data']
        self.current_sr = chop.get('sample_rate', 22050)
        self.current_chop_id = chop_id
        
        # Enable controls
        self.play_btn['state'] = 'normal'
        self.select_btn['state'] = 'normal'
        
        # Visualize waveform
        self.visualize_selected_chop(chop)
        
    def visualize_selected_chop(self, chop):
        """Visualize the selected chop"""
        self.fig_chop.clear()
        ax = self.fig_chop.add_subplot(111)
        
        # Plot waveform
        time = np.linspace(0, len(self.current_audio)/self.current_sr, 
                          len(self.current_audio))
        ax.plot(time, self.current_audio, color=self.colors['accent'], linewidth=0.5)
        ax.fill_between(time, self.current_audio, alpha=0.3, color=self.colors['accent'])
        
        # Add RMS envelope
        rms = librosa.feature.rms(y=self.current_audio)[0]
        rms_time = np.linspace(0, len(self.current_audio)/self.current_sr, len(rms))
        ax.plot(rms_time, rms * 3, color=self.colors['secondary'], 
               linewidth=2, alpha=0.8, label='Energy')
        
        # Styling
        ax.set_title(f"{chop['type']} - {chop['text'][:40]}...", 
                    color=self.colors['fg'])
        ax.set_xlabel("Time (s)", color=self.colors['fg'])
        ax.set_ylabel("Amplitude", color=self.colors['fg'])
        ax.set_facecolor(self.colors['bg'])
        ax.tick_params(colors=self.colors['fg'])
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        self.fig_chop.patch.set_facecolor(self.colors['panel'])
        self.canvas_chop.draw()
        
    def play_chop(self):
        """Play the selected chop"""
        if self.current_audio is not None:
            self.stop_btn['state'] = 'normal'
            sd.play(self.current_audio, self.current_sr)
            self.status_var.set("Playing...")
            
    def stop_playback(self):
        """Stop audio playback"""
        sd.stop()
        self.stop_btn['state'] = 'disabled'
        self.status_var.set("Stopped")
        
    def add_to_selection(self):
        """Add current chop to selection"""
        if hasattr(self, 'current_chop_id'):
            if self.current_chop_id not in self.selected_chops:
                self.selected_chops.append(self.current_chop_id)
                self.selection_var.set(f"{len(self.selected_chops)} chops selected")
                
                # Highlight in tree
                self.chop_tree.item(self.current_chop_id, 
                                   tags=('selected',))
                self.chop_tree.tag_configure('selected', 
                                           background=self.colors['selected'])
                
    def export_selected(self):
        """Export selected chops to a folder"""
        if not self.selected_chops:
            messagebox.showwarning("No Selection", "No chops selected for export")
            return
            
        # Ask for export directory
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        if not export_dir:
            return
            
        export_path = Path(export_dir) / f"chopsuey_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        export_path.mkdir(exist_ok=True)
        
        # Copy selected chops
        for chop_id in self.selected_chops:
            data = self.all_chops[chop_id]
            chop = data['chop']
            metadata = data['metadata']
            
            # Create filename
            filename = f"{chop_id}_{chop['type']}_{metadata['artist']}.wav"
            filepath = export_path / filename
            
            # Save audio
            import soundfile as sf
            sf.write(filepath, chop['audio_data'], 
                    chop.get('sample_rate', 22050))
            
            # Save metadata
            meta_file = export_path / f"{chop_id}_metadata.json"
            with open(meta_file, 'w') as f:
                json.dump({
                    'chop_id': chop_id,
                    'type': chop['type'],
                    'text': chop['text'],
                    'artist': metadata['artist'],
                    'song': metadata['title'],
                    'timestamp': {
                        'start': chop['start'],
                        'end': chop['end']
                    }
                }, f, indent=2)
                
        messagebox.showinfo("Export Complete", 
                           f"Exported {len(self.selected_chops)} chops to:\n{export_path}")
        
    def processing_complete(self):
        """Handle processing completion"""
        self.processing = False
        self.process_btn['state'] = 'normal'
        self.export_btn['state'] = 'normal'
        self.status_var.set(f"Complete! {len(self.all_chops)} chops extracted")
        

class GUIVisualizer:
    """Custom visualizer that sends updates to GUI queue"""
    def __init__(self, queue):
        self.queue = queue
        
    def visualize_chop(self, audio, sr, text, chop_type, features, chop_id=None):
        """Send visualization data to GUI"""
        # For now, just return empty results
        # The GUI handles its own visualization
        return {}


def main():
    """Run the GUI application"""
    root = tk.Tk()
    app = ChopsueyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

