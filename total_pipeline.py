"""
TOTAL PIPELINE — Download → Total Capture → Voice → Transcripts. Autonomous.

Watches demo_downloads/ for new .dem.zst files.
Runs tardigrade_total.py on each.
Extracts voice to WAV.
Transcribes with Whisper.
All automatic, all parallel where possible.
"""
import os
import sys
import time
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

DEMO_DIR = Path(__file__).parent / "demo_downloads"
TOTAL_DIR = Path(__file__).parent / "tardigrade_total_out"
VOICE_DIR = Path(__file__).parent / "voice_output"

TOTAL_DIR.mkdir(exist_ok=True)
VOICE_DIR.mkdir(exist_ok=True)


def extract_voice(dem_path, match_id):
    """Extract voice from demo → WAV files."""
    try:
        import zstandard
        from demoparser2 import DemoParser
        import wave

        data = dem_path.read_bytes()
        if dem_path.suffix == '.zst':
            data = zstandard.ZstdDecompressor().decompress(data)

        tmp = Path(f"/dev/shm/voice_{match_id}.dem")
        tmp.write_bytes(data)
        del data

        parser = DemoParser(str(tmp))
        voice = parser.parse_voice()
        player_info = parser.parse_player_info()
        name_map = {int(r['steamid']): r['name'] for _, r in player_info.iterrows()}

        tmp.unlink()

        if not voice:
            return 0

        import opuslib
        from collections import defaultdict

        player_packets = defaultdict(list)
        for v in voice:
            player_packets[v['steamid']].append(v)

        decoder = opuslib.Decoder(24000, 1)
        match_voice_dir = VOICE_DIR / match_id
        match_voice_dir.mkdir(exist_ok=True)

        decoded_count = 0
        for steamid, packets in player_packets.items():
            name = name_map.get(steamid, str(steamid))
            packets.sort(key=lambda x: x['tick'])

            all_pcm = bytearray()
            for pkt in packets:
                try:
                    pcm = decoder.decode(pkt['bytes'], 480)
                    all_pcm.extend(pcm)
                except:
                    all_pcm.extend(b'\x00' * 960)

            wav_path = match_voice_dir / f'{name}.wav'
            with wave.open(str(wav_path), 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(bytes(all_pcm))

            decoded_count += 1

        return decoded_count

    except Exception as e:
        print(f"    Voice extraction failed: {e}")
        return 0


def transcribe_match(match_id):
    """Transcribe all voice files for a match."""
    try:
        from faster_whisper import WhisperModel

        match_voice_dir = VOICE_DIR / match_id
        if not match_voice_dir.exists():
            return 0

        model = WhisperModel('base', device='cpu', compute_type='int8')
        transcripts = {}

        for wav in sorted(match_voice_dir.glob('*.wav')):
            name = wav.stem
            segments, _ = model.transcribe(str(wav), language='en', vad_filter=True)
            texts = [{'start': round(s.start, 2), 'end': round(s.end, 2),
                      'text': s.text.strip()} for s in segments]
            transcripts[name] = texts

        if transcripts:
            (match_voice_dir / 'transcripts.json').write_text(
                json.dumps(transcripts, indent=2, ensure_ascii=False))

        return sum(len(v) for v in transcripts.values())

    except Exception as e:
        print(f"    Transcription failed: {e}")
        return 0


def process_demo(dem_path):
    """Full pipeline: total capture + voice + transcript."""
    match_id = dem_path.stem.replace('.dem', '')

    # Skip if already done
    total_out = TOTAL_DIR / f"{match_id}_total.json.zst"
    if total_out.exists():
        return None

    t0 = time.time()
    print(f"\n[{match_id}]")

    # 1. Total capture
    print(f"  Total capture...", flush=True)
    result = subprocess.run(
        [sys.executable, 'tardigrade_total.py', str(dem_path), str(TOTAL_DIR)],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[:200]}")
        return None

    # Parse output for stats
    for line in result.stdout.split('\n'):
        if 'columns' in line or 'Events' in line or 'Map' in line:
            print(f"  {line.strip()}")

    # 2. Voice extraction
    print(f"  Voice extraction...", flush=True)
    n_voices = extract_voice(dem_path, match_id)
    print(f"  {n_voices} players' voice decoded")

    # 3. Transcription
    if n_voices > 0:
        print(f"  Transcribing...", flush=True)
        n_segments = transcribe_match(match_id)
        print(f"  {n_segments} transcript segments")

    dt = time.time() - t0
    print(f"  Done in {dt:.0f}s")
    return match_id


def main():
    print("TOTAL PIPELINE — autonomous full extraction")
    print(f"  Watching: {DEMO_DIR}")
    print(f"  Output:   {TOTAL_DIR}")
    print(f"  Voice:    {VOICE_DIR}")
    print()

    while True:
        # Find unprocessed demos
        demos = sorted(DEMO_DIR.glob("*.dem.zst"))
        done = {f.stem.replace('_total.json', '') for f in TOTAL_DIR.glob("*_total.json.zst")}
        todo = [d for d in demos if d.stem.replace('.dem', '') not in done]

        if todo:
            print(f"Found {len(todo)} unprocessed demos")
            for dem in todo:
                try:
                    process_demo(dem)
                except Exception as e:
                    print(f"  ERROR: {e}")
        else:
            print(".", end="", flush=True)

        time.sleep(30)  # Check every 30 seconds


if __name__ == "__main__":
    main()
