"""
A python script that chops a folder of songs into
usable labeled chops for producers.
"""

import os
import zipfile
import shutil
import json
import logging
from pathlib import Path

import librosa
import numpy as np
from demucs.separate import main as demucs_main
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import soundfile as sf
import click
from dotenv import load_dotenv

from chopsuey_visualizer import ChopsueyVisualizer

# Optional imports
try:
    import whisper
except ImportError:
    whisper = None

try:
    import acoustid
except ImportError:
    acoustid = None

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ACOUSTID_API_KEY = os.getenv('ACOUSTID_API_KEY')


class VocalChopExtractor:
    def __init__(self, output_dir="output_chops", visualizer=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.visualizer = visualizer or ChopsueyVisualizer(self.output_dir)
        self.whisper_model = whisper.load_model("base") if whisper else None

    def process_zip(self, zip_path):
        temp_dir = Path("temp_extract")
        temp_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = [m for m in zip_ref.infolist()
                       if not m.filename.startswith('__MACOSX/') and '.DS_Store' not in m.filename]
            zip_ref.extractall(temp_dir, members=members)
        audio_files = [*temp_dir.rglob('*.mp3'), *temp_dir.rglob('*.wav'), *temp_dir.rglob('*.flac')]
        for audio_file in audio_files:
            self.process_file(audio_file)
        shutil.rmtree(temp_dir)

    def process_file(self, audio_path):
        logger.info(f"Processing: {audio_path.name}")
        metadata = self.identify_song(audio_path)
        vocals_path = self.separate_vocals(audio_path)
        transcription = self.transcribe_lyrics(vocals_path)
        # Run all segmentation strategies
        for mode, outdir in [
            ("default", None),
            ("hard", "output_hard"),
            ("harder", "output_harder")
        ]:
            chops = self.analyze_for_chops(vocals_path, transcription, metadata, mode=mode)
            self.save_chops(chops, metadata, audio_path.name, out_dir=outdir)
            if self.visualizer:
                for chop in chops:
                    self.visualizer.visualize_chop(
                        audio=chop['audio_data'],
                        sr=chop['sample_rate'],
                        text=chop['text'],
                        chop_type=chop['type'],
                        features=chop['features'],
                        chop_id=f"{audio_path.stem}_{chop['type']}_{int(chop['start'])}s"
                    )
        # Visual summary
        if self.visualizer:
            self.visualizer.generate_comparison_heatmap()
            self.visualizer.generate_summary_report({
                'input': str(audio_path),
                'output': str(self.output_dir)
            })
            self.visualizer.print_analysis_complete(len(chops))

    def identify_song(self, audio_path):
        if not acoustid or not ACOUSTID_API_KEY:
            logger.warning("AcoustID not available or API key missing.")
            return {'artist': 'Unknown', 'title': Path(audio_path).stem}
        try:
            duration, fingerprint = acoustid.fingerprint_file(audio_path)
            results = acoustid.lookup(ACOUSTID_API_KEY, fingerprint, duration)
            if results and results['results']:
                rec = results['results'][0].get('recordings', [{}])[0]
                return {
                    'artist': rec.get('artists', [{}])[0].get('name', 'Unknown'),
                    'title': rec.get('title', Path(audio_path).stem)
                }
        except Exception as e:
            logger.error(f"Could not identify song: {e}")
        return {'artist': 'Unknown', 'title': Path(audio_path).stem}

    def separate_vocals(self, audio_path):

        output_path = self.output_dir / "separated" / Path(audio_path).stem
        output_path.parent.mkdir(exist_ok=True, parents=True)

        # Run demucs separation
        demucs_main(["-n", "htdemucs", "--two-stems=vocals",
                     "-o", str(output_path.parent), str(audio_path)])

        # This is a placeholder: replace with your actual separation logic
        # Assume vocals.wav is created at output_dir / "vocals.wav"
        # Replace this with your actual separation call
        # vocals_path = output_dir / "vocals.wav"
        vocals_path = output_path.parent / "htdemucs" / Path(audio_path).stem / "vocals.wav"
        if not vocals_path.exists():
            logger.warning(f"Vocals not found at {vocals_path}, using og audio.")
            return audio_path
        # logger.info(f"output_dir  {output_dir}")
        # logger.info(f"outp.parent  {output_dir.parent}")
        # logger.info(f" audio_path  {audio_path}")
        # logger.info(f"vocals_path  {vocals_path}")
        return vocals_path

    def transcribe_lyrics(self, vocals_path):
        if not self.whisper_model:
            logger.warning("Whisper not available, skipping transcription.")
            return {}
        return self.whisper_model.transcribe(
            str(vocals_path),
            language="en",
            fp16=False,
            word_timestamps=True
        )

    def analyze_for_chops(self, vocals_path, transcription, metadata, mode="default"):
        y, sr = librosa.load(vocals_path, sr=None)
        audio = AudioSegment.from_wav(vocals_path)
        # Choose segmentation strategy
        if mode == "harder":
            nonsilent_ranges = detect_nonsilent(audio, min_silence_len=100, silence_thresh=-60, seek_step=5)
            onset_times = librosa.frames_to_time(librosa.onset.onset_detect(y=y, sr=sr, backtrack=True), sr=sr)
            transcription_boundaries = self._get_transcription_boundaries(transcription)
            boundaries = self._combine_boundaries(nonsilent_ranges, onset_times, transcription_boundaries, len(y)/sr)
            boundaries = self._enforce_max_length(boundaries, max_length=8.0)
        elif mode == "hard":
            nonsilent_ranges = detect_nonsilent(audio, min_silence_len=200, silence_thresh=-45, seek_step=10)
            boundaries = self._merge_close_ranges(nonsilent_ranges, 100)
        else:  # default
            nonsilent_ranges = detect_nonsilent(audio, min_silence_len=100, silence_thresh=-40)
            boundaries = [(start/1000, end/1000) for start, end in nonsilent_ranges]
        # Extract chops
        chops = []
        for start_sec, end_sec in boundaries:
            if end_sec - start_sec < 0.1:
                continue
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            segment_audio = y[start_sample:end_sample]
            text = self._get_text_for_timestamp(transcription, start_sec, end_sec)
            features = self._extract_audio_features(segment_audio, sr, text)
            chop_type = self._classify_chop(features, text)
            chops.append({
                'start': start_sec, 'end': end_sec, 'text': text, 'type': chop_type,
                'audio_data': segment_audio, 'sample_rate': sr, 'features': features
            })
        # If too few, force split
        if mode == "harder" and len(chops) < 5:
            chops = self._force_segment_long_chops(chops, y, sr, transcription)
        return chops

    # --- Segmentation helpers ---
    def _merge_close_ranges(self, ranges, max_gap_ms):
        merged = []
        for start, end in ranges:
            if merged and start - merged[-1][1] < max_gap_ms:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return [(s/1000, e/1000) for s, e in merged]

    def _get_transcription_boundaries(self, transcription):
        if 'segments' in transcription:
            return [(seg['start'], seg['end']) for seg in transcription['segments'] if 'start' in seg and 'end' in seg]
        return []

    def _combine_boundaries(self, silence_ranges, onset_times, transcription_boundaries, total_duration):
        all_times = set([0.0, total_duration])
        for start_ms, end_ms in silence_ranges:
            all_times.add(start_ms / 1000)
            all_times.add(end_ms / 1000)
        all_times.update(onset_times)
        for start, end in transcription_boundaries:
            all_times.add(start)
            all_times.add(end)
        sorted_times = sorted(all_times)
        # Create ranges, merge close
        ranges = []
        for i in range(len(sorted_times) - 1):
            start, end = sorted_times[i], sorted_times[i+1]
            if end - start > 0.1:
                ranges.append((start, end))
        merged = []
        for start, end in ranges:
            if merged and start - merged[-1][1] < 0.03:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return merged

    def _enforce_max_length(self, boundaries, max_length=8.0):
        enforced = []
        for start, end in boundaries:
            duration = end - start
            if duration <= max_length:
                enforced.append((start, end))
            else:
                num_splits = int(np.ceil(duration / max_length))
                split_duration = duration / num_splits
                for i in range(num_splits):
                    s = start + i * split_duration
                    e = min(start + (i + 1) * split_duration, end)
                    enforced.append((s, e))
        return enforced

    def _force_segment_long_chops(self, chops, y, sr, transcription):
        new_chops = []
        for chop in chops:
            duration = chop['end'] - chop['start']
            if duration > 16:
                start_sample = int(chop['start'] * sr)
                end_sample = int(chop['end'] * sr)
                segment = y[start_sample:end_sample]
                tempo, beats = librosa.beat.beat_track(y=segment, sr=sr)
                beat_times = librosa.frames_to_time(beats, sr=sr) + chop['start']
                beats_per_chop = 8 if tempo > 120 else 4
                for i in range(0, len(beat_times) - beats_per_chop, beats_per_chop):
                    ns, ne = beat_times[i], beat_times[min(i + beats_per_chop, len(beat_times) - 1)]
                    nsamp, neamp = int(ns * sr), int(ne * sr)
                    new_segment = y[nsamp:neamp]
                    new_text = self._get_text_for_timestamp(transcription, ns, ne)
                    features = self._extract_audio_features(new_segment, sr, new_text)
                    new_type = self._classify_chop(features, new_text)
                    new_chops.append({
                        'start': ns, 'end': ne, 'text': new_text, 'type': new_type,
                        'audio_data': new_segment, 'sample_rate': sr, 'features': features
                    })
            else:
                new_chops.append(chop)
        return new_chops

    # --- Feature extraction and classification ---
    def _extract_audio_features(self, audio, sr, text):
        features = {}
        features['duration'] = len(audio) / sr if sr else 0
        features['mean_energy'] = float(np.mean(np.abs(audio)))
        rms = librosa.feature.rms(y=audio)[0]
        features['energy_variance'] = float(np.var(rms))
        features['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
        pitches, mags = librosa.piptrack(y=audio, sr=sr)
        features['pitch_content'] = float(np.sum(mags > 0.1) / mags.size) if mags.size else 0
        onsets = librosa.onset.onset_detect(y=audio, sr=sr)
        features['onset_density'] = float(len(onsets) / features['duration']) if features['duration'] else 0
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = float(tempo)
        except Exception:
            features['tempo'] = 0.0
        return features

    def _classify_chop(self, features, text):
        d = features['duration']
        e = features['mean_energy']
        od = features['onset_density']
        pc = features['pitch_content']
        ev = features['energy_variance']
        text_upper = text.upper() if text else ""
        if d < 0.5:
            if e > 0.05 and od > 2:
                return "oneshot"
            elif pc < 0.1:
                return "perc_hit"
        elif d < 2:
            if any(word in text_upper for word in ['YEAH', 'OH', 'UH', 'HEY']):
                return "adlib"
            elif pc > 0.3 and od < 4:
                return "vocal_phrase"
            elif od > 4:
                return "vocal_chop"
            else:
                return "ear_candy"
        elif d < 4:
            if ev < 0.005 and pc > 0.2:
                return "vocal_sustain"
            elif od > 3:
                return "vocal_rhythm"
            elif "number one" in text.lower() or any(hook in text.lower() for hook in ['chorus', 'hook']):
                return "hook"
            else:
                return "loop"
        elif d < 8:
            if pc > 0.4 and ev > 0.002:
                return "vocal_melody"
            elif "number one" in text.lower():
                return "chorus_section"
            else:
                return "verse_section"
        else:
            return "full_section"
        return "misc"

    def _get_text_for_timestamp(self, transcription, start, end):
        words = []
        if 'segments' not in transcription:
            return ''
        for segment in transcription['segments']:
            if segment.get('end', 0) < start or segment.get('start', 0) > end:
                continue
            if 'words' in segment and segment['words']:
                for word in segment['words']:
                    if word.get('end', 0) >= start and word.get('start', 0) <= end:
                        words.append(word['word'].strip())
            else:
                if segment.get('start', 0) <= end and segment.get('end', float('inf')) >= start:
                    words.append(segment.get('text', '').strip())
        return ' '.join(words).strip()

    # --- Saving ---
    def save_chops(self, chops, metadata, source_file, out_dir=None):
        safe_artist = self._sanitize_filename(metadata['artist'])
        safe_title = self._sanitize_filename(metadata['title'])
        artist_folder = self.output_dir / f"{safe_artist} - {safe_title}"
        if out_dir:
            out_dir = Path(out_dir)
            out_dir.mkdir(exist_ok=True)
            artist_folder = out_dir / f"{safe_artist} - {safe_title}"
        artist_folder.mkdir(exist_ok=True)
        chop_index = 0
        for chop in chops:
            chop_index += 1
            timestamp = f"{int(chop['start'])}s-{int(chop['end'])}s"
            text_preview = self._sanitize_filename(' '.join(chop['text'].split()[:3]))[:30] if chop['text'] else ''
            filename = f"{chop_index:02d}_{chop['type']}_{timestamp}"
            if text_preview:
                filename += f"_{text_preview}"
            filename += ".wav"
            sample_rate = chop.get('sample_rate', 22050)
            sf.write(artist_folder / filename, chop['audio_data'], sample_rate)
            meta = {
                'artist': metadata['artist'],
                'song': metadata['title'],
                'source_file': str(source_file),
                'timestamp': {'start': chop['start'], 'end': chop['end'], 'duration': chop['end'] - chop['start']},
                'text': chop['text'],
                'type': chop['type'],
                'usage_suggestions': self._get_usage_suggestions(chop['type']),
                'audio_features': chop['features']
            }
            meta_filename = f"{chop_index:02d}_{chop['type']}_metadata.json"
            with open(artist_folder / meta_filename, 'w') as f:
                json.dump(meta, f, indent=2)
        return str(artist_folder)

    def _sanitize_filename(self, text):
        if not text:
            return 'unknown'
        for c in ' /\\:*?"<>|.,;!@#$%^&()[]{}=+~`':
            text = text.replace(c, '_')
        while '__' in text:
            text = text.replace('__', '_')
        return text.strip('_') or 'unknown'

    def _get_usage_suggestions(self, chop_type):
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
@click.option('--inpath', '-i', default='songs.zip', help='Input Zipfolder')
@click.option('--output', '-o', default='output', help='Output Directory')
def main(inpath, output):
    extractor = VocalChopExtractor(output)
    extractor.process_zip(inpath)
    logger.info(f"Complete! Vocal chops saved to: {output}")

if __name__ == '__main__':
    main()
