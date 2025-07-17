"""
chopsuey_visualizer.py - Visualization tools for audio chop analysis

Provides terminal and graphical 
visualization of audio features 
for the Chopsuey project.
"""

import numpy as np
import librosa
import librosa.display
from pathlib import Path
import json
from datetime import datetime
import logging

# Handle optional dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Image visualizations disabled.")

logger = logging.getLogger(__name__)


class ChopsueyVisualizer:
    """Handles all visualization for audio chop analysis"""

    def __init__(self, output_dir="output_chops", verbose=True, save_images=True):
        """
        Initialize the visualizer

        Args:
            output_dir: Base directory for output files
            verbose: Whether to print terminal visualizations
            save_images: Whether to save image visualizations
        """
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.save_images = save_images and MATPLOTLIB_AVAILABLE

        # Create visualization directory
        self.viz_dir = self.output_dir / "visualizations"
        if self.save_images:
            self.viz_dir.mkdir(exist_ok=True, parents=True)

        # Store all chop features for comparison
        self.all_chops_features = []

    def visualize_chop(self, audio, sr, text, chop_type, features, chop_id=None):
        """
        Main method to visualize a single chop

        Args:
            audio: Audio data array
            sr: Sample rate
            text: Transcribed text
            chop_type: Classification type
            features: Dictionary of extracted features
            chop_id: Optional identifier for the chop

        Returns:
            Dict with paths to generated visualizations
        """
        results = {}

        # Terminal visualization
        if self.verbose:
            self._visualize_terminal(audio, sr, text, chop_type, features)

        # Image visualizations
        if self.save_images:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chop_id = chop_id or f"{chop_type}_{timestamp}"

            # Detailed analysis plot
            detail_path = self.viz_dir / f"{chop_id}_analysis.png"
            self._generate_detailed_plot(audio, sr, text, chop_type, features, detail_path)
            results['detail_plot'] = str(detail_path)

            # Feature radar chart
            radar_path = self.viz_dir / f"{chop_id}_features.png"
            self._generate_radar_chart(features, chop_type, radar_path)
            results['radar_chart'] = str(radar_path)

        # Store features for comparison
        self.all_chops_features.append({
            'id': chop_id,
            'type': chop_type,
            'features': features
        })
        
        return results
    
    def _visualize_terminal(self, audio, sr, text, chop_type, features):
        """Create ASCII visualization in terminal"""
        # Header
        print("\n" + "="*60)
        print(f"🎵 CHOP ANALYSIS: {chop_type.upper()}")
        print("="*60)
        
        # Text preview
        text_preview = text[:50] + "..." if len(text) > 50 else text
        print(f"📝 Text: {text_preview}")
        print(f"⏱️  Duration: {features.get('duration', 0):.2f}s")
        
        # Feature bars
        print("\n" + "-"*60)
        self._print_feature_bars(features)
        
        # Waveform
        print("\n" + "-"*60)
        self._print_waveform_ascii(audio, sr)
        
        # Classification confidence
        if 'classification_confidence' in features:
            print(f"\n🎯 Classification Confidence: {features['classification_confidence']:.1%}")
        
        print("="*60 + "\n")
    
    def _print_feature_bars(self, features):
        """Print ASCII bar charts for features"""
        # Define feature ranges and symbols
        feature_configs = {
            'Energy': {
                'value': features.get('mean_energy', 0) * 100,
                'symbol': '█',
                'color_code': '\033[91m'  # Red
            },
            'Dynamics': {
                'value': features.get('energy_variance', 0) * 200,
                'symbol': '▓',
                'color_code': '\033[93m'  # Yellow
            },
            'Brightness': {
                'value': (features.get('spectral_centroid', 0) / 10000) * 30,
                'symbol': '▒',
                'color_code': '\033[96m'  # Cyan
            },
            'Pitchiness': {
                'value': features.get('pitch_content', 0) * 30,
                'symbol': '♪',
                'color_code': '\033[92m'  # Green
            },
            'Rhythmic': {
                'value': min(features.get('onset_density', 0) * 3, 30),
                'symbol': '♫',
                'color_code': '\033[95m'  # Magenta
            }
        }

        print("AUDIO FEATURES:")
        for name, config in feature_configs.items():
            value = int(config['value'])
            bar = config['symbol'] * value

            # Add color if terminal supports it
            if self._terminal_supports_color():
                bar = f"{config['color_code']}{bar}\033[0m"

            raw_value = features.get(name.lower().replace(' ', '_'), 0)
            print(f"  {name:12} {bar} ({raw_value:.3f})")

#     def _print_waveform_ascii(self, audio, sr, width=60, height=7):
#         """ASCII waveform visualization"""
#         # Downsample for display
#         hop = max(len(audio) // width, 1)

#         simplified = []
#         for i in range(0, len(audio), hop):
#             window = audio[i:i+hop]
#             if len(window) > 0:
#                 simplified.append(np.mean(np.abs(window)))

#         # Normalize
#         max_val = max(simplified) if simplified else 1
#         if max_val > 0:
#             normalized = [int((v/max_val) * height) for v in simplified]
#         else:
#             normalized = [0] * len(simplified)

#         # Print waveform
#         print("WAVEFORM:")
#         for h in range(height, -1, -1):
#             line = ""
#             for v in normalized[:width]:
#                 if v >= h:
#                     line += "█"
#                 else:
#                     line += " "
#             print(f"  |{line}")
#         print(f"  +{'-'*width}")
#         print(f"   0{'s':>{width-2}}{features.get('duration', 0):.1f}s")

    def _print_waveform_ascii(self, audio, sr, width=60, height=7):
        """ASCII waveform visualization"""
        # Calculate duration from audio and sample rate
        duration = len(audio) / sr

        # Downsample for display
        hop = max(len(audio) // width, 1)

        simplified = []
        for i in range(0, len(audio), hop):
            window = audio[i:i+hop]
            if len(window) > 0:
                simplified.append(np.mean(np.abs(window)))

        # Normalize
        max_val = max(simplified) if simplified else 1
        if max_val > 0:
            normalized = [int((v/max_val) * height) for v in simplified]
        else:
            normalized = [0] * len(simplified)

        # Print waveform
        print("WAVEFORM:")
        for h in range(height, -1, -1):
            line = ""
            for v in normalized[:width]:
                if v >= h:
                    line += "█"
                else:
                    line += " "
            print(f"  |{line}")
        print(f"  +{'-'*width}")

        # Time axis with proper formatting
        duration_str = f"{duration:.1f}s"
        print(f"   0s{duration_str:>{width-2}}")

    def _generate_detailed_plot(self, audio, sr, text, chop_type, features, output_path):
        """Generate detailed analysis plot"""
        if not MATPLOTLIB_AVAILABLE:
            return
    
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(4, 2, height_ratios=[1.5, 1, 1, 1])
        
        # Title without emoji
        fig.suptitle(f'Audio Chop Analysis: {chop_type.upper()}', fontsize=16)
        
        # 1. Waveform with energy envelope
        ax1 = plt.subplot(gs[0, :])
        time = np.linspace(0, len(audio)/sr, len(audio))
        ax1.plot(time, audio, color='cyan', linewidth=0.5, alpha=0.7)
        ax1.fill_between(time, audio, alpha=0.3, color='cyan')
        
        # Add RMS energy envelope
        rms = librosa.feature.rms(y=audio)[0]
        rms_time = np.linspace(0, len(audio)/sr, len(rms))
        ax1.plot(rms_time, rms * 3, color='orange', linewidth=2, label='Energy Envelope')
        
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Waveform & Energy - Duration: {features.get("duration", 0):.2f}s')
        ax1.set_xlim(0, len(audio)/sr)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Spectrogram
        ax2 = plt.subplot(gs[1, :])
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax2, cmap='magma')
        ax2.set_title('Frequency Content (Spectrogram)')
        fig.colorbar(img, ax=ax2, format='%+2.0f dB')
        
        # 3. Spectral features over time
        ax3 = plt.subplot(gs[2, 0])
        cent = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        cent_time = np.linspace(0, len(audio)/sr, len(cent))
        
        ax3.plot(cent_time, cent, color='yellow', linewidth=2, label='Centroid')
        ax3.plot(cent_time, rolloff, color='orange', linewidth=2, label='Rolloff')
        ax3.axhline(y=features.get('spectral_centroid', 0), color='red', 
                    linestyle='--', alpha=0.5, label='Mean Centroid')
        ax3.set_ylabel('Frequency (Hz)')
        ax3.set_title('Spectral Features')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Onset detection
        ax4 = plt.subplot(gs[2, 1])
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Plot onset strength
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onset_time = np.linspace(0, len(audio)/sr, len(onset_env))
        ax4.plot(onset_time, onset_env, color='green', linewidth=2)
        ax4.vlines(onset_times, 0, onset_env.max(), color='red', alpha=0.9, 
                linestyle='--', label=f'Onsets ({len(onset_times)})')
        ax4.set_ylabel('Onset Strength')
        ax4.set_title(f'Rhythm Analysis - Density: {features.get("onset_density", 0):.1f}/s')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Feature summary (without emojis)
        ax5 = plt.subplot(gs[3, :])
        ax5.axis('off')
        
        # Create feature summary text without emojis
        summary = f"""
        TEXT: {text[:80]}{'...' if len(text) > 80 else ''}
        
        CLASSIFICATION: {chop_type.upper()}
        
        KEY METRICS:
        • Duration: {features.get('duration', 0):.2f}s
        • Mean Energy: {features.get('mean_energy', 0):.3f}
        • Energy Variance: {features.get('energy_variance', 0):.4f}
        • Spectral Centroid: {features.get('spectral_centroid', 0):.0f} Hz
        • Pitch Content: {features.get('pitch_content', 0):.2%}
        • Onset Density: {features.get('onset_density', 0):.1f}/s
        • Tempo: {features.get('tempo', 'N/A')} BPM
        
        USAGE SUGGESTIONS: {', '.join(features.get('usage_suggestions', ['general use']))}
        """
        
        ax5.text(0.05, 0.95, summary, transform=ax5.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=1", facecolor="gray", alpha=0.2))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()

#     def emoji_generate_detailed_plot(self, audio, sr, text, chop_type, features, output_path):
#         """Generate detailed analysis plot"""
#         if not MATPLOTLIB_AVAILABLE:
#             return
# 
#         plt.style.use('dark_background')
#         fig = plt.figure(figsize=(14, 10))
#         gs = gridspec.GridSpec(4, 2, height_ratios=[1.5, 1, 1, 1])
# 
#         # Title with emoji
#         type_emojis = {
#             'oneshot': '💥', 'drop': '🎯', 'drone': '🌊', 'ear_candy': '✨',
#             'vocal_phrase': '🎤', 'vocal_sustain': '🎵', 'vocal_rhythm': '🥁',
#             'loop': '🔄', 'vocal_melody': '🎼', 'atmosphere': '🌌',
#             'perc_hit': '🥁', 'adlib': '🗣️', 'hook': '🎣'
#         }
#         emoji = type_emojis.get(chop_type, '🎵')
#         fig.suptitle(f'{emoji} Audio Chop Analysis: {chop_type.upper()}', fontsize=16)
# 
#         # 1. Waveform with energy envelope
#         ax1 = plt.subplot(gs[0, :])
#         time = np.linspace(0, len(audio)/sr, len(audio))
#         ax1.plot(time, audio, color='cyan', linewidth=0.5, alpha=0.7)
#         ax1.fill_between(time, audio, alpha=0.3, color='cyan')
# 
#         # Add RMS energy envelope
#         rms = librosa.feature.rms(y=audio)[0]
#         rms_time = np.linspace(0, len(audio)/sr, len(rms))
#         ax1.plot(rms_time, rms * 3, color='orange', linewidth=2, label='Energy Envelope')
# 
#         ax1.set_ylabel('Amplitude')
#         ax1.set_title(f'Waveform & Energy - Duration: {features.get("duration", 0):.2f}s')
#         ax1.set_xlim(0, len(audio)/sr)
#         ax1.legend()
#         ax1.grid(True, alpha=0.3)
# 
#         # 2. Spectrogram
#         ax2 = plt.subplot(gs[1, :])
#         D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
#         img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax2, cmap='magma')
#         ax2.set_title('Frequency Content (Spectrogram)')
#         fig.colorbar(img, ax=ax2, format='%+2.0f dB')
# 
#         # 3. Spectral features over time
#         ax3 = plt.subplot(gs[2, 0])
#         cent = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
#         rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
#         cent_time = np.linspace(0, len(audio)/sr, len(cent))
# 
#         ax3.plot(cent_time, cent, color='yellow', linewidth=2, label='Centroid')
#         ax3.plot(cent_time, rolloff, color='orange', linewidth=2, label='Rolloff')
#         ax3.axhline(y=features.get('spectral_centroid', 0), color='red', 
#                     linestyle='--', alpha=0.5, label='Mean Centroid')
#         ax3.set_ylabel('Frequency (Hz)')
#         ax3.set_title('Spectral Features')
#         ax3.legend()
#         ax3.grid(True, alpha=0.3)
# 
#         # 4. Onset detection
#         ax4 = plt.subplot(gs[2, 1])
#         onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
#         onset_times = librosa.frames_to_time(onset_frames, sr=sr)
# 
#         # Plot onset strength
#         onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
#         onset_time = np.linspace(0, len(audio)/sr, len(onset_env))
#         ax4.plot(onset_time, onset_env, color='green', linewidth=2)
#         ax4.vlines(onset_times, 0, onset_env.max(), color='red', alpha=0.9, 
#                    linestyle='--', label=f'Onsets ({len(onset_times)})')
#         ax4.set_ylabel('Onset Strength')
#         ax4.set_title(f'Rhythm Analysis - Density: {features.get("onset_density", 0):.1f}/s')
#         ax4.legend()
#         ax4.grid(True, alpha=0.3)
# 
#         # 5. Feature summary
#         ax5 = plt.subplot(gs[3, :])
#         ax5.axis('off')
# 
#         # Create feature summary text
#         summary = f"""
#         📝 TEXT: {text[:80]}{'...' if len(text) > 80 else ''}
# 
#         🎯 CLASSIFICATION: {chop_type.upper()}
# 
#         📊 KEY METRICS:
#         • Duration: {features.get('duration', 0):.2f}s
#         • Mean Energy: {features.get('mean_energy', 0):.3f}
#         • Energy Variance: {features.get('energy_variance', 0):.4f}
#         • Spectral Centroid: {features.get('spectral_centroid', 0):.0f} Hz
#         • Pitch Content: {features.get('pitch_content', 0):.2%}
#         • Onset Density: {features.get('onset_density', 0):.1f}/s
#         • Tempo: {features.get('tempo', 'N/A')} BPM
#         
#         💡 USAGE SUGGESTIONS: {', '.join(features.get('usage_suggestions', ['general use']))}
#         """
#         
#         ax5.text(0.05, 0.95, summary, transform=ax5.transAxes, 
#                  fontsize=10, verticalalignment='top',
#                  bbox=dict(boxstyle="round,pad=1", facecolor="gray", alpha=0.2))
#         
#         plt.tight_layout()
#         plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
#         plt.close()
    
    def _generate_radar_chart(self, features, chop_type, output_path):
        """Generate radar chart of features"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Define categories and values
        categories = ['Energy', 'Dynamics', 'Brightness', 'Pitchiness', 'Rhythmic']
        values = [
            min(features.get('mean_energy', 0) * 5, 1),
            min(features.get('energy_variance', 0) * 50, 1),
            min(features.get('spectral_centroid', 0) / 8000, 1),
            features.get('pitch_content', 0),
            min(features.get('onset_density', 0) / 10, 1)
        ]
        
        # Complete the circle
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        # Plot
        ax.plot(angles, values, 'o-', linewidth=2, color='lime', markersize=8)
        ax.fill(angles, values, alpha=0.25, color='lime')
        
        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12)
        ax.set_ylim(0, 1)
        ax.set_title(f'{chop_type.upper()} - Feature Profile', size=16, pad=20)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for angle, value, cat in zip(angles[:-1], values[:-1], categories):
            ax.text(angle, value + 0.05, f'{value:.2f}', 
                    ha='center', va='center', size=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def generate_comparison_heatmap(self, output_filename="chops_comparison.png"):
        """Generate heatmap comparing all analyzed chops"""
        if not MATPLOTLIB_AVAILABLE or not self.all_chops_features:
            return None
        
        output_path = self.viz_dir / output_filename
        
        # Prepare data
        feature_names = ['Duration', 'Energy', 'Dynamics', 'Brightness', 'Pitch', 'Rhythm']
        chop_labels = []
        matrix = []
        
        for i, chop_data in enumerate(self.all_chops_features):
            chop = chop_data['features']
            chop_type = chop_data['type']
            chop_labels.append(f"{i+1:02d}_{chop_type}")
            
            # Normalize features to 0-1 range
            matrix.append([
                min(chop.get('duration', 0) / 10, 1),
                min(chop.get('mean_energy', 0) * 5, 1),
                min(chop.get('energy_variance', 0) * 50, 1),
                min(chop.get('spectral_centroid', 0) / 8000, 1),
                chop.get('pitch_content', 0),
                min(chop.get('onset_density', 0) / 10, 1)
            ])
        
        # Create figure
        fig_height = max(6, len(chop_labels) * 0.3 + 2)
        plt.figure(figsize=(10, fig_height))
        
        # Create heatmap
        sns.heatmap(matrix, 
                    xticklabels=feature_names,
                    yticklabels=chop_labels,
                    cmap='viridis',
                    cbar_kws={'label': 'Normalized Value (0-1)'},
                    annot=True,
                    fmt='.2f',
                    linewidths=0.5)
        
        plt.title('Audio Chops Feature Comparison', fontsize=16)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Chops', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    def generate_summary_report(self, metadata, output_filename="analysis_summary.json"):
        """Generate a JSON summary of all analyzed chops"""
        summary = {
            'metadata': metadata,
            'analysis_date': datetime.now().isoformat(),
            'total_chops': len(self.all_chops_features),
            'chop_types': {},
            'average_features': {},
            'chops': []
        }
        
        # Count chop types
        for chop_data in self.all_chops_features:
            chop_type = chop_data['type']
            summary['chop_types'][chop_type] = summary['chop_types'].get(chop_type, 0) + 1
        
        # Calculate average features
        if self.all_chops_features:
            feature_sums = {}
            feature_counts = {}
            
            for chop_data in self.all_chops_features:
                for key, value in chop_data['features'].items():
                    if isinstance(value, (int, float)):
                        feature_sums[key] = feature_sums.get(key, 0) + value
                        feature_counts[key] = feature_counts.get(key, 0) + 1
            
            for key in feature_sums:
                summary['average_features'][key] = feature_sums[key] / feature_counts[key]
        
        # Add individual chop data
        summary['chops'] = self.all_chops_features
        
        # Save to file
        output_path = self.output_dir / output_filename
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return str(output_path)
    
    def _terminal_supports_color(self):
        """Check if terminal supports color"""
        import sys
        import os
        
        # Windows
        if sys.platform == 'win32':
            return os.environ.get('ANSICON') is not None
        
        # Unix/Linux/Mac
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    
    def print_analysis_complete(self, total_chops):
        """Print completion message with summary"""
        if not self.verbose:
            return
        
        print("\n" + "="*60)
        print("🎉 ANALYSIS COMPLETE!")
        print("="*60)
        print(f"Total chops analyzed: {total_chops}")
        
        if self.all_chops_features:
            print("\nChop type distribution:")
            type_counts = {}
            for chop in self.all_chops_features:
                chop_type = chop['type']
                type_counts[chop_type] = type_counts.get(chop_type, 0) + 1
            
            for chop_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * count
                print(f"  {chop_type:15} {bar} ({count})")
        
        if self.save_images:
            print(f"\nVisualizations saved to: {self.viz_dir}")
        
        print("="*60 + "\n")


# Example usage function
def example_usage():
    """Example of how to use the visualizer"""
    # Create visualizer instance
    visualizer = ChopsueyVisualizer(
        output_dir="output_chops",
        verbose=True,
        save_images=True
    )
    
    # Example audio and features
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t) * np.exp(-t)  # Decaying sine wave
    
    features = {
        'duration': duration,
        'mean_energy': 0.3,
        'energy_variance': 0.01,
        'spectral_centroid': 440,
        'pitch_content': 0.8,
        'onset_density': 2.0,
        'tempo': 120,
        'usage_suggestions': ['melodic', 'tonal']
    }
    
    # Visualize
    viz_paths = visualizer.visualize_chop(
        audio=audio,
        sr=sr,
        text="Example vocal phrase",
        chop_type="vocal_phrase",
        features=features,
        chop_id="example_001"
    )
    
    print(f"Generated visualizations: {viz_paths}")


if __name__ == "__main__":
    example_usage()

