"""
    A python script that chops a folder of songs into
    usable labled chops for producers.

"""
# In chopsuey.py, add this import
from chopsuey_visualizer import ChopsueyVisualizer

import zipfile
import os
import json
from pathlib import Path
import librosa
import click
import numpy as np
from demucs.separate import main as demucs_main
import whisper
import acoustid
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import shutil
import soundfile as sf
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API key from environment variables (optional)
ACOUSTID_API_KEY = os.getenv('ACOUSTID_API_KEY')


class VocalChopExtractor:
    """

    Provides functions needed of song id and chopping.

    """
    def __init__(self, output_dir="output_chops"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.whisper_model = whisper.load_model("base")

        # Initialize visualizer
        self.visualizer = ChopsueyVisualizer(
            output_dir="output",
            verbose=True,
            save_images=True
        )

    def print_chop(self, sr=22050, duration=2.0,
                   mean=0.3, var=0.01, spec=440,
                   pitch=0.8, onset=2.0, tempo=120,
                   usage= ['melodic', 'tonal'],
                   text="Example Vocal",
                   chop_type="vocal phrase",
                   chop_id="example_001"):
        """
        Print a Chop

        """
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
        viz_paths = self.visualizer.visualize_chop(
            audio=audio,
            sr=sr,
            text="Example vocal phrase",
            chop_type="vocal_phrase",
            features=features,
            chop_id="example_001"
        )

        print(f"Generated visualizations: {viz_paths}")

    def identify_song(self, audio_path):
        """

        Identify song using AcoustID

        """
        if not ACOUSTID_API_KEY:
            logger.Warning("Warning: AcoustID_API_KEY not set")
            return {'artist': 'Unknown', 'title': Path(audio_path).stem}

        try:
            duration, fingerprint = acoustid.fingerprint_file(audio_path)
            results = acoustid.lookup("MwTy7IdhRm", fingerprint, duration)

            if results and results['results']:
                result = results['results'][0]
                recordings = result.get('recordings', [])
                if recordings:
                    recording = recordings[0]
                    return {
                            'artist': recording['artists'][0]['name'],
                            'title': recording['title']
                            }
        except Exception as e:
            logger.error(f"Could not identify song: {e}")

        return {'artist': 'Unknown', 'title': Path(audio_path).stem}

    def separate_vocals(self, audio_path):
        """Separate vocals using demucs"""
        # logger.info(f"  {audio_path}")
        output_path = self.output_dir / "separated" / Path(audio_path).stem

        # Run demucs separation
        demucs_main(["-n", "htdemucs", "--two-stems=vocals",
                     "-o", str(output_path.parent), str(audio_path)])

        # logger.info(f"output_path  {output_path}")
        # logger.info(f"outp.parent  {output_path.parent}")
        # logger.info(f" audio_path  {audio_path}")
        vocals_path = output_path.parent / "htdemucs" / Path(audio_path).stem / "vocals.wav"
        # logger.info(f"vocals_path  {vocals_path}")
        return vocals_path

    def transcribe_lyrics(self, vocals_path):
        """Transcribe lyrics using Whisper"""
        result = self.whisper_model.transcribe(
                str(vocals_path),
                language="en",
                fp16=False,
                word_timestamps=True)
        return result

    def analyze_for_chops(self, vocals_path, transcription, metadata):
        """Analyze vocals for interesting chops"""
        y, sr = librosa.load(vocals_path)

        # Detect non-silent segments
        audio = AudioSegment.from_wav(vocals_path)
        nonsilent_ranges = detect_nonsilent(audio,
                                            min_silence_len=100,
                                            silence_thresh=-40)

        chops = []

        # Analyze each segment
        for start_ms, end_ms in nonsilent_ranges:
            start_sec = start_ms / 1000
            end_sec = end_ms / 1000

            # Find corresponding text
            text = self._get_text_for_timestamp(transcription, start_sec, end_sec)

            # Analyze audio characteristics
            segment_audio = y[int(start_sec * sr):int(end_sec * sr)]

            # Check for notable characteristics
            chop_type = self._classify_chop(segment_audio, sr, text)

            if chop_type:
                chops.append({
                    'start': start_sec,
                    'end': end_sec,
                    'text': text,
                    'type': chop_type,
                    'audio_data': segment_audio
                })

        return chops

    def analyze_for_chops_hard(self, vocals_path, transcription, metadata):
        """Analyze vocals for interesting chops with better segmentation"""
        y, sr = librosa.load(vocals_path, sr=None)  # Keep original sample rate

        # Use more sophisticated silence detection
        audio = AudioSegment.from_wav(vocals_path)

        # Adjust silence detection parameters based on content
        nonsilent_ranges = detect_nonsilent(
                audio,
                min_silence_len=200,  # Increased from 100ms
                silence_thresh=-45,    # More sensitive
                seek_step=10          # More precise
                )

        # Merge very close segments (within 100ms)
        merged_ranges = []
        for i, (start_ms, end_ms) in enumerate(nonsilent_ranges):
            if merged_ranges and start_ms - merged_ranges[-1][1] < 100:
                # Merge with previous
                merged_ranges[-1] = (merged_ranges[-1][0], end_ms)
            else:
                merged_ranges.append((start_ms, end_ms))

        chops = []

        for start_ms, end_ms in merged_ranges:
            start_sec = start_ms / 1000
            end_sec = end_ms / 1000
            duration = end_sec - start_sec

            # Skip very short segments (likely noise)
            if duration < 0.1:
                continue

            # Get audio segment
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            segment_audio = y[start_sample:end_sample]

            # Get corresponding text
            text = self._get_text_for_timestamp(transcription, start_sec, end_sec)

            # Classify chop
            chop_type = self._classify_chop(segment_audio, sr, text)

            # Extract features for visualization
            features = self._extract_audio_features(segment_audio, sr)
            features['duration'] = duration
            features['usage_suggestions'] = self._get_usage_suggestions(chop_type)
            # Visualize if enabled
            if self.visualizer:
                chop_id = f"{int(start_sec)}s_{chop_type}"
                viz_results = self.visualizer.visualize_chop(
                    audio=segment_audio,
                    sr=sr,
                    text=text,
                    chop_type=chop_type,
                    features=features,
                    chop_id=chop_id
                )

                # Store chop with visualization info
                chops.append({
                    'start': start_sec,
                    'end': end_sec,
                    'text': text,
                    'type': chop_type,
                    'audio_data': segment_audio,
                    'sample_rate': sr,
                    'features': features,
                    'visualizations': viz_results if self.visualizer else {}
                })

        return chops

    # In your analyze_for_chops method:
    def analyze_for_chops_harder(self, vocals_path, transcription, metadata):
        """Analyze vocals for interesting chops with multiple strategies"""
        y, sr = librosa.load(vocals_path, sr=None)

        # Strategy 1: Silence-based detection with refined parameters
        audio = AudioSegment.from_wav(vocals_path)

        # More aggressive silence detection for continuous vocals
        nonsilent_ranges = detect_nonsilent(
                audio,
                min_silence_len=100,    # Reduced from 200ms - catch smaller gaps
                silence_thresh=-60,     # More sensitive to find quieter sections
                seek_step=5            # More precise scanning
                )

        # Strategy 2: Add onset detection for additional segmentation points
        onset_frames = librosa.onset.onset_detect(
                y=y, sr=sr,
                backtrack=True,
                pre_max=20,
                post_max=20,
                pre_avg=100,
                post_avg=100,
                delta=0.2,
                wait=10
                )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        # Strategy 3: Use transcription boundaries if available
        transcription_boundaries = self._get_transcription_boundaries(transcription)

        # Combine all segmentation points
        all_boundaries = self._combine_boundaries(
                nonsilent_ranges, onset_times, transcription_boundaries, len(y)/sr
                )

        # Apply maximum chop length
        all_boundaries = self._enforce_max_length(all_boundaries, max_length=8.0)

        chops = []

        for start_sec, end_sec in all_boundaries:
            duration = end_sec - start_sec

            # Skip very short segments
            if duration < 0.15:
                continue

            # Get audio segment
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)

            # Ensure bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(y), end_sample)

            if start_sample >= end_sample:
                continue

            segment_audio = y[start_sample:end_sample]

            # Get corresponding text
            text = self._get_text_for_timestamp(transcription, start_sec, end_sec)

            # Classify chop
            chop_type = self._classify_chop_hard(segment_audio, sr, text)

            chops.append({
                'start': start_sec,
                'end': end_sec,
                'text': text,
                'type': chop_type,
                'audio_data': segment_audio,
                'sample_rate': sr
                })

        # If we still have very few chops, force-split long segments
        if len(chops) < 5:
            logger.info("Few chops detected, applying force segmentation...")
            chops = self._force_segment_long_chops_hard(chops, y, sr, transcription)

        return chops

    def _get_transcription_boundaries(self, transcription):
        """Extract natural boundaries from transcription"""
        boundaries = []

        if 'segments' in transcription:
            for segment in transcription['segments']:
                if 'start' in segment and 'end' in segment:
                    boundaries.append((segment['start'], segment['end']))

        return boundaries

    def _combine_boundaries(self, silence_ranges, onset_times, transcription_boundaries, total_duration):
        """Combine different boundary detection methods"""
        all_times = set([0.0, total_duration])

        # Add silence-based boundaries
        for start_ms, end_ms in silence_ranges:
            all_times.add(start_ms / 1000)
            all_times.add(end_ms / 1000)

        # Add onset times
        all_times.update(onset_times)

        # Add transcription boundaries
        for start, end in transcription_boundaries:
            all_times.add(start)
            all_times.add(end)

        # Sort and create ranges
        sorted_times = sorted(all_times)
        ranges = []

        for i in range(len(sorted_times) - 1):
            start = sorted_times[i]
            end = sorted_times[i + 1]

            # Only create range if it's meaningful in size
            if end - start > 0.1:  # At least 100ms
                ranges.append((start, end))

        # Merge very close segments (within 30ms instead of 100ms)
        merged_ranges = []
        for start, end in ranges:
            if merged_ranges and start - merged_ranges[-1][1] < 0.03:
                merged_ranges[-1] = (merged_ranges[-1][0], end)
            else:
                merged_ranges.append((start, end))

        return merged_ranges

    def _enforce_max_length(self, boundaries, max_length=8.0):
        """Split any segments longer than max_length"""
        enforced_boundaries = []

        for start, end in boundaries:
            duration = end - start

            if duration <= max_length:
                enforced_boundaries.append((start, end))
            else:
                # Split long segments into roughly equal parts
                num_splits = int(np.ceil(duration / max_length))
                split_duration = duration / num_splits

                for i in range(num_splits):
                    split_start = start + (i * split_duration)
                    split_end = min(start + ((i + 1) * split_duration), end)
                    enforced_boundaries.append((split_start, split_end))

        return enforced_boundaries

    def _force_segment_long_chops(self, chops, y, sr, transcription):
        """Force segmentation of long chops when too few are detected"""
        new_chops = []

        for chop in chops:
            duration = chop['end'] - chop['start']

            if duration > 16:  # If longer than 16 seconds
                # Try to split based on beat tracking
                start_sample = int(chop['start'] * sr)
                end_sample = int(chop['end'] * sr)
                segment = y[start_sample:end_sample]

                # Detect beats in this segment
                tempo, beats = librosa.beat.beat_track(y=segment, sr=sr)
                beat_times = librosa.frames_to_time(beats, sr=sr)

                # Add offset for original position
                beat_times = beat_times + chop['start']

                # Create chops at every 4 or 8 beats
                beats_per_chop = 8 if tempo > 120 else 4

                for i in range(0, len(beat_times) - beats_per_chop, beats_per_chop):
                    new_start = beat_times[i]
                    new_end = beat_times[min(i + beats_per_chop, len(beat_times) - 1)]

                    # Get the audio for this segment
                    new_start_sample = int(new_start * sr)
                    new_end_sample = int(new_end * sr)
                    new_segment = y[new_start_sample:new_end_sample]

                    # Get text
                    new_text = self._get_text_for_timestamp(transcription, new_start, new_end)

                    # Reclassify
                    new_type = self._classify_chop_hard(new_segment, sr, new_text)

                    new_chops.append({
                        'start': new_start,
                        'end': new_end,
                        'text': new_text,
                        'type': new_type,
                        'audio_data': new_segment,
                        'sample_rate': sr
                        })
            else:
                new_chops.append(chop)

        return new_chops

    def _classify_chop_hard(self, audio, sr, text):
        """Enhanced classification with better handling of continuous vocals"""
        duration = len(audio) / sr
        logger.info(f"duration {duration}")

        # Calculate features
        rms_energy = librosa.feature.rms(y=audio)[0]
        mean_energy = np.mean(rms_energy)
        energy_variance = np.var(rms_energy)
        logger.info(f"rms_enery {rms_energy}")
        logger.info(f"mean_energy {mean_energy}")
        logger.info(f"energy_variance {energy_variance}")

        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        logger.info(f"spectral_centroid {spectral_centroid}")
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        logger.info(f"zcr {zcr}")

        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        logger.info(f"pitches, magnitudes {pitches} {magnitudes}")
        pitch_content = np.sum(magnitudes > 0.1) / magnitudes.size

        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
        logger.info(f"onset_frames {onset_frames}")
        onset_density = len(onset_frames) / duration if duration > 0 else 0
        logger.info(f"onset_density {onset_density}")

        # More nuanced classification
        if duration < 0.5:
            if mean_energy > 0.05 and onset_density > 2:
                return "oneshot"
            elif pitch_content < 0.1:
                return "perc_hit"

        elif duration < 2:
            if any(word in text.upper() for word in [
                            'YEAH', 'OH', 'UH', 'HEY']):
                return "adlib"
            elif pitch_content > 0.3 and onset_density < 4:
                return "vocal_phrase"
            elif onset_density > 4:
                return "vocal_chop"
            else:
                return "ear_candy"

        elif duration < 4:
            if energy_variance < 0.005 and pitch_content > 0.2:
                return "vocal_sustain"
            elif onset_density > 3:
                return "vocal_rhythm"
            elif "number one" in text.lower() or any(hook in text.lower() for hook in ['chorus', 'hook']):
                return "hook"
            else:
                return "loop"

        elif duration < 8:
            if pitch_content > 0.4 and energy_variance > 0.002:
                return "vocal_melody"
            elif "number one" in text.lower():
                return "chorus_section"
            else:
                return "verse_section"

        else:  # 8+ seconds - these should be rare with max length enforcement
            return "full_section"

        return "misc"

    def _classify_chop(self, audio, sr, text):
        """Classify the type of vocal chop with improved analysis"""
        # Calculate multiple audio features
        duration = len(audio) / sr

        # Energy and dynamics
        rms_energy = librosa.feature.rms(y=audio)[0]
        mean_energy = np.mean(rms_energy)
        energy_variance = np.var(rms_energy)

        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

        # Pitch detection for vocal content
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_content = np.sum(magnitudes > 0.1) / magnitudes.size

        # Onset detection for rhythmic content
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
        onset_density = len(onset_frames) / duration if duration > 0 else 0

        # Improved classification logic
        if duration < 0.5:
            if mean_energy > 0.05 and onset_density > 2:
                return "oneshot"
            elif pitch_content < 0.1:  # Percussive
                return "perc_hit"

        elif duration < 1.5:
            # Check for exclamatory or impactful content
            if (mean_energy > 0.08 and energy_variance > 0.01) or \
                    (text and any(char in text.upper() for char in [
                        '!', '?', 'YEAH', 'OH', 'UH', 'HEY'])):
                return "drop"
            elif pitch_content > 0.3:  # Clear vocal content
                return "vocal_phrase"
            else:
                return "ear_candy"

        elif duration < 4:
            if energy_variance < 0.005 and pitch_content > 0.2:  # Sustaine voc
                return "vocal_sustain"
            elif onset_density > 3:  # Rhythmic pattern
                return "vocal_rhythm"
            else:
                return "loop"

        else:  # duration >= 4
            if energy_variance < 0.003:  # Very steady
                return "drone"
            elif pitch_content > 0.4:  # Melodic content
                return "vocal_melody"
            else:
                return "atmosphere"

        return "misc"  # Default category

    def _get_text_for_timestamp(self, transcription, start, end):
        """Extract text for given timestamp range - improved version"""
        words = []

        if 'segments' not in transcription:
            return ''

        for segment in transcription['segments']:
            # First check segment-level overlap
            if 'start' in segment and 'end' in segment:
                # Skip segments that don't overlap with our range
                if segment['end'] < start or segment['start'] > end:
                    continue

            # Check word-level if available
            if 'words' in segment and segment['words']:
                for word in segment['words']:
                    if 'start' in word and 'end' in word:
                        # Include words that overlap with the time range at all
                        word_start = word['start']
                        word_end = word['end']

                        # Check if word overlaps with our range
                        if word_end >= start and word_start <= end:
                            words.append(word['word'].strip())
            else:
                # Fallback to segment text if no word-level timestamps
                if segment.get('start', 0) <= end and segment.get('end', float('inf')) >= start:
                    words.append(segment.get('text', '').strip())

        return ' '.join(words).strip()


    def save_chops(self, chops, metadata, source_file, out_dir=None):
        """Save vocal chops with improved naming"""
        # Create artist folder
        safe_artist = self._sanitize_filename(metadata['artist'])
        safe_title = self._sanitize_filename(metadata['title'])
        artist_folder = self.output_dir / f"{safe_artist} - {safe_title}"
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(exist_ok=True)
            artist_folder = out_dir / f"{safe_artist} - {safe_title}"

        logger.info(f"artist_folder {artist_folder}")
        artist_folder.mkdir(exist_ok=True)
        # Group chops by type for better organization
        chops_by_type = {}
        for chop in chops:
            chop_type = chop['type']
            if chop_type not in chops_by_type:
                chops_by_type[chop_type] = []
            chops_by_type[chop_type].append(chop)

        # Save chops
        chop_index = 0
        for chop_type, typed_chops in chops_by_type.items():
            for i, chop in enumerate(typed_chops):
                chop_index += 1

                # Create descriptive filename
                timestamp = f"{int(chop['start'])}s-{int(chop['end'])}s"

                # Better text handling
                text_preview = ''
                if chop['text']:
                    # Take first few words, clean them
                    words = chop['text'].split()[:3]  
                    # First 3 words
                    text_preview = '_'.join(words)
                    # Limit length
                    text_preview = self._sanitize_filename(text_preview)[:30]  

                filename = f"{chop_index:02d}_{chop_type}_{timestamp}"
                if text_preview:
                    filename += f"_{text_preview}"
                filename += ".wav"

                # Save audio with original sample rate
                sample_rate = chop.get('sample_rate', 22050)
                sf.write(
                        artist_folder / filename, 
                        chop['audio_data'], 
                        sample_rate
                        )

                # Enhanced metadata
                meta = {
                        'artist': metadata['artist'],
                        'song': metadata['title'],
                        'source_file': str(source_file),
                        'timestamp': {
                            'start': chop['start'],
                            'end': chop['end'],
                            'duration': chop['end'] - chop['start']
                            },
                        'text': chop['text'],
                        'type': chop['type'],
                        'usage_suggestions': self._get_usage_suggestions(chop['type']),
                        'audio_features': self._extract_audio_features(chop['audio_data'], sample_rate)
                        }

                meta_filename = f"{chop_index:02d}_{chop_type}_metadata.json"
                with open(artist_folder / meta_filename, 'w') as f:
                    json.dump(meta, f, indent=2)

        return str(artist_folder)

    def _sanitize_filename(self, text):
        """Properly sanitize text for filenames"""
        if not text:
            return 'unknown'

        # Replace problematic characters
        replacements = {
                ' ': '_',
                '/': '_',
                '\\': '_',
                ':': '_',
                '*': '_',
                '?': '_',
                '"': '_',
                '<': '_',
                '>': '_',
                '|': '_',
                '.': '_',
                ',': '_',
                ';': '_',
                '!': '_',
                '@': '_',
                '#': '_',
                '$': '_',
                '%': '_',
                '^': '_',
                '&': '_',
                '(': '_',
                ')': '_',
                '[': '_',
                ']': '_',
                '{': '_',
                '}': '_',
                '=': '_',
                '+': '_',
                '~': '_',
                '`': '_'
                }

        result = text
        for old, new in replacements.items():
            result = result.replace(old, new)

        # Remove multiple underscores
        while '__' in result:
            result = result.replace('__', '_')

        # Remove leading/trailing underscores
        result = result.strip('_')

        return result if result else 'unknown'

    def _extract_audio_features(self, audio, sr):
        """Extract additional audio features for metadata"""
        features = {}

        try:
            # Basic features
            features['duration'] = len(audio) / sr
            features['mean_amplitude'] = float(np.mean(np.abs(audio)))
            features['max_amplitude'] = float(np.max(np.abs(audio)))

            # Spectral features
            features['spectral_centroid'] = float(
                    np.mean(
                        librosa.feature.spectral_centroid(y=audio, sr=sr)))
            features['spectral_rolloff'] = float(
                    np.mean(
                        librosa.feature.spectral_rolloff(y=audio, sr=sr)))

            # Tempo estimation for rhythmic content
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)

            # Tempo estimation for rhythmic content
            try:
                tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
                # Safely extract tempo value
                if hasattr(tempo, 'item'):
                    features['estimated_tempo'] = float(tempo.item())
                elif hasattr(tempo, '__len__') and len(tempo) > 0:
                    features['estimated_tempo'] = float(tempo[0])
                else:
                    features['estimated_tempo'] = float(tempo) if tempo is not None else 0.0
            except Exception as e:
                features['estimated_tempo'] = 0.0
                logger.error(f"Could not estimate tempo: {e}")

        except Exception as e:
            logger.error(f"Could not extract all audio features: {e}")

        return features

    def _get_usage_suggestions(self, chop_type):
        """Enhanced usage suggestions based on chop type"""
        suggestions = {
                'oneshot': ['percussive hit', 'transition', 'accent', 'stab'],
                'drop': ['section marker', 'impact', 'emphasis', 'buildup end'],
                'drone': ['atmosphere', 'texture', 'pad layer', 'background'],
                'ear_candy': ['fill', 'variation', 'interest point', 'fx'],
                'vocal_phrase': ['hook', 'topline', 'ad-lib', 'vocal sample'],
                'vocal_sustain': ['texture', 'harmony layer', 'ambient vocal'],
                'vocal_rhythm': ['percussive vocal', 'beatbox element', 'rhythm layer'],
                'loop': ['4-bar loop', '8-bar loop', 'groove element'],
                'vocal_melody': ['melodic hook', 'topline', 'vocal lead'],
                'atmosphere': ['intro/outro', 'breakdown', 'ambient section'],
                'perc_hit': ['kick layer', 'snare layer', 'percussion'],
                'misc': ['general sample', 'creative use']
                }
        return suggestions.get(chop_type, ['general sample'])

@click.command()
@click.option('--inpath', '-i', 'inpath', default='songs.zip',
              help='Input Zipfolder')
@click.option('--output', '-o', 'outpath', default='output',
              help='Output Directory')
def main(inpath='songs.zip', outpath='output'):
    """
        Extract vocal chops from a ZIP file of audio tracks
    """
# In your VocalChopExtractor class __init__:
# After processing all files:
    # logger.info(f"inpath {inpath}")
    # logger.info(f"outpath {outpath}")
    extractor = VocalChopExtractor(outpath)

    temp_dir = Path('temp_extract')
    temp_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(inpath, 'r') as zip_ref:

        # Get list of members to extract
        # (excluding __MACOSX and .DS_Store anywhere in path)
        members_to_extract = [
                member for member in zip_ref.infolist()
                if not member.filename.startswith('__MACOSX/')
                and '.DS_Store' not in member.filename
                ]

        # Extract all at once
        zip_ref.extractall(temp_dir, members=members_to_extract)

    audio_files = list(temp_dir.rglob('*.mp3')) + \
            list(temp_dir.rglob('*.wav')) + \
            list(temp_dir.rglob('*.flac'))

    # logger.info(f"audio_files {audio_files}")

    for audio_file in audio_files:
        logger.info(f"Processing: {audio_file.name}")

        # Identify song
        metadata = extractor.identify_song(audio_file)
        logger.info(f"  Identified as: {metadata['artist']} - "
                    f"{metadata['title']}")

        logger.info(f" Full Metadata  {metadata}")
        # Separate vocals
        logger.info("  Separating vocals...")
        vocals_path = extractor.separate_vocals(audio_file)
        logger.info(f" Vocals Separated to {vocals_path}")

        # Transcribe
        logger.info("  Transcribing lyrics...")
        transcr = extractor.transcribe_lyrics(vocals_path)
        # logger.debug(f"  Transcription {transcr}")

        # Analyze for chops
        logger.info("  Analyzing for vocal chops...")
        chops = extractor.analyze_for_chops(vocals_path, transcr, metadata)
        # logger.info(f"  Chops {chops}")
        logger.info(f"  Found {len(chops)} interesting chops")

        logger.info("  HARD MODE Analyzing for vocal chops...")
        chops_hard = extractor.analyze_for_chops_hard(vocals_path, transcr, metadata)
        #  logger.info(f"  Chops Hard {chops_hard}")
        logger.info(f"  Found {len(chops_hard)} interesting chops")

        logger.info("  HARDER MODE Analyzing for vocal chops...")
        chops_harder = extractor.analyze_for_chops_harder(vocals_path, transcr, metadata)
        logger.info(f"  Chop Harder {chops_harder}")

        # Save results
        extractor.save_chops(chops, metadata, audio_file.name)
        extractor.save_chops(chops_hard, metadata, audio_file.name, 'output_hard')
        extractor.save_chops(chops_harder, metadata, audio_file.name, 'output_harder')

        # After processing all files
        if extractor.visualizer:
            # Generate comparison heatmap
            heatmap_path = extractor.visualizer.generate_comparison_heatmap()
            if heatmap_path:
                logger.info(f"Generated comparison heatmap: {heatmap_path}")

            summary_path = extractor.visualizer.generate_summary_report({
                'input': inpath,
                'output': outpath,
                'processed_files': len(audio_files)
            })
            logger.info(f"Generated analysis summary: {summary_path}")

    # Cleanup
    shutil.rmtree(temp_dir)

    # Print completion message
    logger.info(f"\nComplete! Vocal chops saved to: {outpath}")


if __name__ == '__main__':
    main()
