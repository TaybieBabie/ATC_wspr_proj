#analyze.py
import json
import os
import re
from dataclasses import asdict
from datetime import datetime

import pandas as pd

from utils.config import TRANSCRIPT_DIR, ANALYSIS_DIR
from utils.atc_utils import CALLSIGN_REGEX
from .transmission import Transmission

# ATC communication patterns
ALTITUDE_PATTERN = r'(\d{1,3},?\d{3})\s*(?:feet|ft)'
HEADING_PATTERN = r'heading\s+(\d{1,3})'
FREQUENCY_PATTERN = r'(\d{3}\.\d{1,3})'


def extract_atc_info(text):
    """Extract ATC-specific information from transcript"""
    info = {
        'callsigns': CALLSIGN_REGEX.findall(text.upper()),
        'altitudes': [int(a.replace(',', '')) for a in re.findall(ALTITUDE_PATTERN, text)],
        'headings': [int(h) for h in re.findall(HEADING_PATTERN, text)],
        'frequencies': re.findall(FREQUENCY_PATTERN, text)
    }
    return info


def analyze_transcript(transcript_file):
    """Analyze a single transcript file"""
    with open(transcript_file, 'r') as f:
        data = json.load(f)

    full_text = data['text']
    segments = data['segments']

    # Extract ATC information
    atc_info = extract_atc_info(full_text)

    # Create segment-level analysis
    transmissions: list[Transmission] = []
    segment_data = []
    for segment in segments:
        segment_text = segment['text']
        segment_info = extract_atc_info(segment_text)
        transmission = Transmission(
            timestamp=datetime.fromtimestamp(segment['start']),
            text=segment_text,
            callsigns=segment_info['callsigns'],
            altitudes=segment_info['altitudes'],
            headings=segment_info['headings'],
            frequencies=segment_info['frequencies']
        )
        transmissions.append(transmission)
        segment_data.append(asdict(transmission))

    return {
        'full_text': full_text,
        'overall_info': atc_info,
        'segments': segment_data
    }, transmissions


def batch_analyze(directory=TRANSCRIPT_DIR):
    """Analyze all transcript files in directory"""
    if not os.path.exists(ANALYSIS_DIR):
        os.makedirs(ANALYSIS_DIR)

    results = []
    for file in os.listdir(directory):
        if file.endswith("_transcript.json"):
            transcript_path = os.path.join(directory, file)
            analysis, _ = analyze_transcript(transcript_path)

            # Save individual analysis
            analysis_file = os.path.join(ANALYSIS_DIR, file.replace('_transcript.json', '_analysis.json'))
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=4)

            results.append({
                'file': file,
                'callsigns': len(analysis['overall_info']['callsigns']),
                'altitudes': len(analysis['overall_info']['altitudes']),
                'headings': len(analysis['overall_info']['headings']),
                'frequencies': len(analysis['overall_info']['frequencies']),
            })

    # Create summary DataFrame
    df = pd.DataFrame(results)
    summary_path = os.path.join(ANALYSIS_DIR, 'analysis_summary.csv')
    df.to_csv(summary_path, index=False)
    print(f"Analysis summary saved to {summary_path}")

    return df


if __name__ == "__main__":
    summary = batch_analyze()
    print("\nAnalysis Summary:")
    print(summary)
