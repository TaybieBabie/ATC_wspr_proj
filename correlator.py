import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from adsb_tracker import ADSBTracker, Aircraft


class ATCCorrelator:
    """Correlates ATC transcripts with ADS-B data"""

    def __init__(self, adsb_tracker: ADSBTracker):
        self.adsb_tracker = adsb_tracker
        self.correlation_window = 30  # seconds to look back/forward

        # Patterns for extracting info from transcripts
        self.patterns = {
            'callsign': re.compile(r'\b([A-Z]{3}[A-Z]?\d{1,4}[A-Z]?)\b'),
            'altitude': re.compile(r'\b(\d{1,3}(?:,\d{3})?)\s*(?:feet|ft)?\b'),
            'heading': re.compile(r'\b(?:heading|turn)\s*(\d{3})\b'),
            'position': re.compile(r'\b(\d{1,2})\s*(?:miles?|nm)\s*(\w+)\b'),
            'altitude_thousands': re.compile(r'\b(?:flight level|FL|altitude)\s*(\d{2,3})\b'),
            'squawk': re.compile(r'\b(?:squawk|transponder)\s*(\d{4})\b'),
        }

    def extract_flight_info(self, transcript: str) -> Dict:
        """Extract flight information from transcript"""
        info = {
            'callsigns': [],
            'altitudes': [],
            'positions': [],
            'headings': [],
            'squawks': []
        }

        # Extract callsigns
        callsigns = self.patterns['callsign'].findall(transcript.upper())
        info['callsigns'] = list(set(callsigns))

        # Extract altitudes
        altitudes = self.patterns['altitude'].findall(transcript)
        for alt in altitudes:
            alt_num = int(alt.replace(',', ''))
            info['altitudes'].append(alt_num)

        # Extract FL altitudes
        fl_altitudes = self.patterns['altitude_thousands'].findall(transcript)
        for fl in fl_altitudes:
            info['altitudes'].append(int(fl) * 100)

        # Extract positions (distance and direction)
        positions = self.patterns['position'].findall(transcript)
        for dist, direction in positions:
            info['positions'].append({
                'distance': int(dist),
                'direction': direction.upper()
            })

        # Extract headings
        headings = self.patterns['heading'].findall(transcript)
        info['headings'] = [int(h) for h in headings]

        # Extract squawk codes
        squawks = self.patterns['squawk'].findall(transcript)
        info['squawks'] = squawks

        return info

    def correlate_transcript(self, transcript: str,
                             timestamp: datetime) -> Dict:
        """Correlate transcript with current ADS-B data"""
        # Update ADS-B data
        self.adsb_tracker.update_aircraft_positions()

        # Extract flight info from transcript
        flight_info = self.extract_flight_info(transcript)

        # Correlation results
        results = {
            'timestamp': timestamp,
            'transcript': transcript,
            'extracted_info': flight_info,
            'correlated_aircraft': [],
            'uncorrelated_callsigns': [],
            'primary_targets': [],
            'alerts': []
        }

        # Check for primary target mentions
        primary_keywords = ['primary target', 'primary only', 'no transponder',
                            'radar contact', 'unidentified']
        transcript_lower = transcript.lower()
        for keyword in primary_keywords:
            if keyword in transcript_lower:
                results['primary_targets'].append({
                    'keyword': keyword,
                    'context': self._extract_context(transcript, keyword)
                })

        # Correlate callsigns
        for callsign in flight_info['callsigns']:
            aircraft = self.adsb_tracker.find_aircraft_by_callsign(callsign)
            if aircraft:
                results['correlated_aircraft'].append({
                    'callsign': callsign,
                    'aircraft': aircraft,
                    'match_type': 'callsign'
                })
            else:
                results['uncorrelated_callsigns'].append(callsign)

                # Check if altitude was mentioned with this callsign
                context = self._extract_callsign_context(transcript, callsign)
                altitude_match = self._find_altitude_in_context(context)

                if altitude_match:
                    # Look for aircraft at that altitude
                    possible_aircraft = self.adsb_tracker.get_aircraft_at_altitude(
                        altitude_match
                    )
                    if possible_aircraft:
                        results['alerts'].append({
                            'type': 'possible_non_transponder',
                            'callsign': callsign,
                            'altitude': altitude_match,
                            'possible_aircraft': possible_aircraft,
                            'confidence': 'medium'
                        })
                    else:
                        results['alerts'].append({
                            'type': 'untracked_aircraft',
                            'callsign': callsign,
                            'altitude': altitude_match,
                            'confidence': 'high'
                        })

        # Check for VFR/1200 squawks
        for squawk in flight_info['squawks']:
            if squawk == '1200':
                results['alerts'].append({
                    'type': 'vfr_aircraft',
                    'squawk': squawk,
                    'confidence': 'high'
                })

        # Analyze uncorrelated callsigns
        if results['uncorrelated_callsigns']:
            self._analyze_uncorrelated(results)

        return results

    def _extract_context(self, text: str, keyword: str,
                         context_chars: int = 100) -> str:
        """Extract context around keyword"""
        text_lower = text.lower()
        pos = text_lower.find(keyword.lower())
        if pos >= 0:
            start = max(0, pos - context_chars)
            end = min(len(text), pos + len(keyword) + context_chars)
            return text[start:end]
        return ""

    def _extract_callsign_context(self, text: str, callsign: str,
                                  context_chars: int = 50) -> str:
        """Extract context around callsign mention"""
        pos = text.upper().find(callsign.upper())
        if pos >= 0:
            start = max(0, pos - context_chars)
            end = min(len(text), pos + len(callsign) + context_chars)
            return text[start:end]
        return ""

    def _find_altitude_in_context(self, context: str) -> Optional[int]:
        """Find altitude mentioned near callsign"""
        alt_matches = self.patterns['altitude'].findall(context)
        if alt_matches:
            return int(alt_matches[0].replace(',', ''))

        fl_matches = self.patterns['altitude_thousands'].findall(context)
        if fl_matches:
            return int(fl_matches[0]) * 100

        return None

    def _analyze_uncorrelated(self, results: Dict):
        """Further analysis of uncorrelated callsigns"""
        for callsign in results['uncorrelated_callsigns']:
            # Check if it's a known non-transponder pattern
            if any(pattern in callsign for pattern in ['NORDO', 'PRIMARY']):
                results['alerts'].append({
                    'type': 'known_non_transponder',
                    'callsign': callsign,
                    'confidence': 'high'
                })