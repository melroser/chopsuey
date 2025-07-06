import click
import zipfile
import os
import json
from pathlib import Path
import librosa
import numpy as np
from demucs.separate import main as demucs_main
import whisper
import acoustid
import requests
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

class VocalChopExtractor:
    def __init__(self, output_dir="output_chops"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.whisper_model = whisper.load_model("base")

    def identify_song(self, audio_path):
        """Identify song using AcoustID"""
        try:
            # Get API key from a placeholder
            api_key = ''
            # Generate fingerprint
            duration, fingerprint = acoustid.fingerprint_file(audio_path)
            
            # Make API request with timeout
            results = acoustid.lookup(api_key, fingerprint, duration, meta=['recordings', 'releases'])
            
            if not results or 'results' not in results or not results['results']:
                print("No results found from AcoustID")
                return {'artist': 'Unknown', 'title': Path(audio_path).stem}
                
            # Get the best match (first result with recordings)
            for result in results['results']:
                if 'recordings' in result and result['recordings']:
                    recording = result['recordings'][0]
                    artist = recording['artists'][0]['name'] if recording.get('artists') else 'Unknown Artist'
                    title = recording.get('title', 'Unknown Title')
                    return {
                        'artist': artist,
                        'title': title
                    }
            
            print("No matching recordings found")
            return {'artist': 'Unknown', 'title': Path(audio_path).stem}
            
        except acoustid.NoBackendError:
            print("Error: Chromaprint library not found")
        except acoustid.FingerprintGenerationError as e:
            print(f"Error generating fingerprint: {e}")
        except acoustid.WebServiceError as e:
            print(f"AcoustID service error: {e}")
        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
        except Exception as e:
            print(f"Unexpected error identifying song: {e}")
        
        return {'artist': 'Unknown', 'title': Path(audio_path).stem}

    def separate_vocals(self, audio_path):
        """Separate vocals using demucs"""
        try:
            # Convert to Path object if it's a string
            audio_path = Path(audio_path)
            
            # Create a safe output directory name
            safe_name = ''.join(c if c.isalnum() or c in ' -_' else '_' for c in audio_path.stem)
            output_dir = self.output_dir / "separated" / safe_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare the output directory for demucs
            demucs_output_dir = self.output_dir / "separated"
            
            # Run demucs separation with proper path handling
            demucs_main([
                "-n", "htdemucs",
                "--two-stems=vocals",
                "-o", str(demucs_output_dir),
                str(audio_path.absolute())
            ])

            # Find the vocals file in the output directory
            # Demucs creates a structure like: output_dir/htdemucs/track_name/vocals.wav
            vocals_dir = demucs_output_dir / "htdemucs" / audio_path.stem
            vocals_path = next(vocals_dir.glob("*vocals.*"), None)
            
            if not vocals_path or not vocals_path.exists():
                # Try alternative path pattern
                vocals_path = vocals_dir / "vocals.wav"
                if not vocals_path.exists():
                    raise FileNotFoundError(
                        f"Could not find separated vocals in {vocals_dir}. "
                        f"Contents: {list(vocals_dir.glob('*')) if vocals_dir.exists() else 'Directory not found'}"
                    )
                
            return str(vocals_path.absolute())
            
        except Exception as e:
            print(f"Error during vocal separation: {e}")
            if 'vocals_dir' in locals():
                print(f"Contents of {vocals_dir}:")
                if vocals_dir.exists():
                    print(list(vocals_dir.glob('*')))
                else:
                    print("Directory does not exist")
            raise

    def transcribe_lyrics(self, audio_path):
        """Transcribe lyrics using Whisper"""
        try:
            # Ensure the file exists and is accessible
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
                
            # Load Whisper model with appropriate settings for CPU
            self.whisper_model = whisper.load_model(
                "base",
                device="cpu",
                download_root=str(Path.home() / ".cache/whisper")
            )
            
            # Transcribe with error handling
            result = self.whisper_model.transcribe(
                str(audio_path),
                language="en",
                word_timestamps=True,
                fp16=False  # Disable FP16 on CPU
            )
            return result
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            raise

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

    def _classify_chop(self, audio, sr, text):
        """Classify the type of vocal chop"""
        # Analyze energy, pitch variation, duration
        energy = np.mean(librosa.feature.rms(y=audio))
        duration = len(audio) / sr

        # Simple classification logic
        if duration < 0.5 and energy > 0.1:
            return "oneshot"
        elif duration < 2 and "!" in text or "?" in text:
            return "drop"
        elif duration > 3:
            return "drone"
        elif energy > 0.15:
            return "ear_candy"

        return None

    def _get_text_for_timestamp(self, transcription, start, end):
        """Extract text for given timestamp range"""
        words = []
        for segment in transcription['segments']:
            for word in segment.get('words', []):
                if start <= word['start'] <= end:
                    words.append(word['word'])
        return ' '.join(words).strip()

    def save_chops(self, chops, metadata, source_file):
        """Save vocal chops with metadata"""
        artist_folder = self.output_dir / f"{metadata['artist']} - {metadata['title']}"
        artist_folder.mkdir(exist_ok=True)

        for i, chop in enumerate(chops):
            timestamp = f"{int(chop['start'])}s-{int(chop['end'])}s"
            safe_text = chop['text'][:20].replace(' ', '_').replace('/', '_')

            filename = f"{i+1}_{timestamp}_{chop['type']}_{safe_text}.wav"

            # Save audio
            import soundfile as sf
            sf.write(artist_folder / filename, chop['audio_data'], 22050)

            # Save metadata
            meta = {
                    'artist': metadata['artist'],
                    'song': metadata['title'],
                    'source_file': source_file,
                    'timestamp': timestamp,
                    'duration': chop['end'] - chop['start'],
                    'text': chop['text'],
                    'type': chop['type'],
                    'usage_suggestions': self._get_usage_suggestions(chop['type'])
                    }

            with open(artist_folder / f"{i+1}_metadata.json", 'w') as f:
                json.dump(meta, f, indent=2)

    def _get_usage_suggestions(self, chop_type):
        """Suggest usage based on chop type"""
        suggestions = {
                'oneshot': ['percussive hit', 'transition', 'accent'],
                'drop': ['section marker', 'impact', 'emphasis'],
                'drone': ['atmosphere', 'texture', 'pad layer'],
                'ear_candy': ['fill', 'variation', 'interest point']
                }
        return suggestions.get(chop_type, ['general sample'])

@click.command()
@click.argument('zip_path', type=click.Path(exists=True))
@click.option('--output', '-o', default='output_chops', help='Output directory')
def main(zip_path, output):
    """Extract vocal chops from a ZIP file of audio tracks"""
    extractor = VocalChopExtractor(output)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        temp_dir = Path('temp_extract')
        temp_dir.mkdir(exist_ok=True)
        zip_ref.extractall(temp_dir)

        # Get all audio files, excluding system files and __MACOSX directory
        audio_files = []
        for ext in ['*.mp3', '*.wav', '*.flac']:
            audio_files.extend([f for f in temp_dir.rglob(ext) 
                             if not any(part.startswith(('.', '_')) for part in f.parts) and 
                             not f.parent.name == '__MACOSX'])

        for audio_file in audio_files:
            try:
                click.echo(f"\nProcessing: {audio_file.name}")

                # Skip system files and hidden files
                if any(part.startswith(('.', '_')) for part in audio_file.parts):
                    click.echo(f"  Skipping system file: {audio_file}")
                    continue

                # Skip files in __MACOSX directory
                if "__MACOSX" in str(audio_file):
                    click.echo(f"  Skipping macOS system file: {audio_file}")
                    continue

                # Identify song
                try:
                    metadata = extractor.identify_song(audio_file)
                    click.echo(f"  Identified as: {metadata['artist']} - {metadata['title']}")
                except Exception as e:
                    click.echo(f"  Could not identify song: {e}")
                    metadata = {'artist': 'Unknown', 'title': audio_file.stem}

                # Separate vocals
                click.echo("  Separating vocals...")
                try:
                    vocals_path = extractor.separate_vocals(audio_file)
                    
                    # Transcribe
                    click.echo("  Transcribing lyrics...")
                    transcription = extractor.transcribe_lyrics(vocals_path)

                    # Analyze for chops
                    click.echo("  Analyzing for vocal chops...")
                    chops = extractor.analyze_for_chops(vocals_path, transcription, metadata)

                    # Save results
                    click.echo(f"  Found {len(chops)} interesting chops")
                    extractor.save_chops(chops, metadata, str(audio_file))
                    
                except Exception as e:
                    click.echo(f"  Error processing {audio_file.name}: {str(e)}", err=True)
                    continue
                    
            except Exception as e:
                click.echo(f"  Unexpected error processing {audio_file.name}: {str(e)}", err=True)
                continue

    # Clean up temporary files
    try:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        click.echo(f"Warning: Could not clean up temporary files: {e}")
        
if __name__ == '__main__':
    main()
