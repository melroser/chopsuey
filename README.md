# Chopsuey

A Python tool for creating usable vocal chops from a folder of songs.

## Prerequisites

- Python 3.8 or higher
- FFmpeg (required for audio processing)
- An AcoustID API key (get one from [acoustid.org/api-key](https://acoustid.org/api-key))

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chopsuey.git
   cd chopsuey
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - Copy the example environment file:
     ```bash
     cp .env.example .env
     ```
   - Edit the `.env` file and add your AcoustID API key:
     ```
     ACOUSTID_API_KEY=your_acoustid_api_key_here
     ```
   - **IMPORTANT**: Never commit your `.env` file to version control.

## Usage

```bash
# Process a single audio file
python chopsuey.py path/to/your/audio.mp3

# Process all audio files in a directory
python chopsuey.py path/to/audio/directory/

# Process a zip file containing audio files
python chopsuey.py path/to/audio/files.zip
```

## How It Works

Chopsuey processes audio files to extract vocal chops by:
1. Separating vocals from the music using Demucs
2. Identifying song information using AcoustID
3. Detecting vocal segments using Whisper
4. Exporting individual vocal chops

## License

MIT
