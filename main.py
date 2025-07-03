import argparse
import os
from record_audio import record_audio
from transcribe import transcribe_audio
from analyze import analyze_transcript
from config import ATC_FREQUENCY


def main():
    parser = argparse.ArgumentParser(description='ATC Communication Transcription and Analysis')
    parser.add_argument('--record', type=int, help='Record audio for specified duration in seconds')
    parser.add_argument('--transcribe', type=str, help='Transcribe specified audio file')
    parser.add_argument('--analyze', type=str, help='Analyze specified transcript file')
    parser.add_argument('--pipeline', type=int, help='Run full pipeline: record, transcribe, and analyze')

    args = parser.parse_args()

    if args.record:
        audio_file = record_audio(duration=args.record, frequency=ATC_FREQUENCY)
    elif args.pipeline:
        print("Running full pipeline...")
        audio_file = record_audio(duration=args.pipeline, frequency=ATC_FREQUENCY)
        transcript = transcribe_audio(audio_file)
        transcript_file = os.path.join('transcripts', os.path.basename(audio_file).replace('.wav', '_transcript.json'))
        analysis = analyze_transcript(transcript_file)
        print("\nTranscription:")
        print(transcript['text'])
        print("\nAnalysis:")
        print(f"Detected {len(analysis['overall_info']['callsigns'])} callsigns")
        print(f"Detected {len(analysis['overall_info']['altitudes'])} altitude instructions")
    elif args.transcribe:
        transcribe_audio(args.transcribe)
    elif args.analyze:
        analyze_transcript(args.analyze)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()