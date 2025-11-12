"""Integration helpers for performing ADS-B/ATC correlation via Ollama LLM."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Sequence

import requests

from tracking.adsb_tracker import Aircraft


@dataclass
class ADSBContact:
    """Lightweight representation of an ADS-B contact for the LLM prompt."""

    icao: str
    callsign: Optional[str]
    altitude: int
    heading: int
    speed: int
    lat: float
    lon: float
    squawk: Optional[str]
    timestamp: float

    def update_position(self, elapsed_seconds: float) -> None:
        """Advance the contact using a simple dead-reckoning model."""
        if elapsed_seconds <= 0:
            return

        nm_per_second = self.speed / 3600.0
        deg_per_nm = 1 / 60.0
        delta = nm_per_second * elapsed_seconds * deg_per_nm

        self.lat += delta * math.cos(math.radians(self.heading))
        self.lon += delta * math.sin(math.radians(self.heading))
        self.timestamp += elapsed_seconds

    @classmethod
    def from_aircraft(cls, aircraft: Aircraft) -> "ADSBContact":
        """Create a contact from an :class:`Aircraft` instance."""
        timestamp = (
            aircraft.timestamp.timestamp()
            if isinstance(aircraft.timestamp, datetime)
            else time.time()
        )
        callsign = aircraft.callsign.strip() or None
        return cls(
            icao=(aircraft.icao24 or "").upper(),
            callsign=callsign,
            altitude=int(aircraft.altitude or 0),
            heading=int(aircraft.track or 0),
            speed=int(aircraft.ground_speed or 0),
            lat=float(aircraft.latitude or 0.0),
            lon=float(aircraft.longitude or 0.0),
            squawk=None,
            timestamp=timestamp,
        )


@dataclass
class ATCTransmission:
    """Summary of an ATC transmission provided to the LLM."""

    timestamp: float
    text: str
    frequency: str
    channel_name: str
    segments: Sequence[dict]
    audio_duration: float
    transcription_delay: float


class ContextBuilder:
    """Builds the prompt used to query the LLM."""

    def __init__(self, max_adsb_contacts: int = 50, max_transmissions: int = 10):
        self.max_adsb = max_adsb_contacts
        self.max_tx = max_transmissions

    def build_prompt(
        self,
        adsb_data: Sequence[ADSBContact],
        transmissions: Sequence[ATCTransmission],
    ) -> str:
        adsb_summary = self._format_adsb(adsb_data[-self.max_adsb :])
        tx_summary = self._format_transmissions(transmissions[-self.max_tx :])

        return f"""You are an aviation ATC analyst. Analyze the following ADS-B radar contacts and ATC radio transmissions to identify correlations.

IMPORTANT TASKS:
1. Match each transmission to an aircraft in the ADS-B data if possible
2. Flag transmissions that reference aircraft NOT in ADS-B data (NON_TRANSPONDER)
3. Flag any military callsigns (MILITARY) - patterns: REACH, VIPER, EAGLE, HAMMER, KING, RESCUE, RCH
4. Handle partial callsigns (e.g., "Bravo 4" might match "N654B4")
5. Handle garbled/unclear transcriptions appropriately

CURRENT ADS-B CONTACTS:
{adsb_summary}

RECENT ATC TRANSMISSIONS:
{tx_summary}

For each transmission, provide:
1. matched_icao: The ICAO hex code if found, or "NO_MATCH" if aircraft not in ADS-B data, or "UNCLEAR" if transmission is garbled
2. confidence: Float 0.0-1.0
3. reasoning: Brief explanation of your matching logic
4. flags: Array containing any of: "MILITARY", "NON_TRANSPONDER", "GARBLED", "PARTIAL_MATCH"

Respond ONLY with valid JSON in this exact format:
{{
  "correlations": [
    {{
      "transmission_id": <index number>,
      "matched_icao": "<ICAO or NO_MATCH or UNCLEAR>",
      "matched_callsign": "<callsign if matched>",
      "confidence": <float>,
      "reasoning": "<explanation>",
      "flags": ["<flag>"]
    }}
  ],
  "alerts": [
    {{
      "type": "<MILITARY|NON_TRANSPONDER|UNKNOWN_TRAFFIC>",
      "details": "<description>",
      "severity": "<HIGH|MEDIUM|LOW>"
    }}
  ],
  "summary": "<brief overall assessment>"
}}"""

    def _format_adsb(self, contacts: Sequence[ADSBContact]) -> str:
        lines: List[str] = []
        for contact in contacts:
            callsign = contact.callsign or "NO_CALLSIGN"
            squawk = contact.squawk or "----"
            lines.append(
                f"ICAO:{contact.icao} CALLSIGN:{callsign:10} "
                f"ALT:{contact.altitude:5}ft HDG:{contact.heading:03}° "
                f"SPD:{contact.speed:3}kt SQUAWK:{squawk}"
            )
        return "\n".join(lines) if lines else "(no ADS-B contacts)"

    def _format_transmissions(self, txs: Sequence[ATCTransmission]) -> str:
        lines: List[str] = []
        for idx, tx in enumerate(txs):
            lines.append(
                f"[{idx}] [{tx.channel_name} {tx.frequency}MHz] {tx.text}"
            )
        return "\n".join(lines) if lines else "(no recent transmissions)"


class OllamaCorrelator:
    """Query an Ollama-hosted model to correlate ATC transmissions."""

    def __init__(
        self,
        model: str = "gpt-oss:20b",
        base_url: str = "http://localhost:11434",
        max_adsb_contacts: int = 50,
        max_transmissions: int = 10,
        request_timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.context_builder = ContextBuilder(
            max_adsb_contacts=max_adsb_contacts,
            max_transmissions=max_transmissions,
        )
        self.request_timeout = request_timeout

    def correlate(
        self,
        adsb_data: Sequence[ADSBContact],
        transmissions: Sequence[ATCTransmission],
    ) -> dict:
        prompt = self.context_builder.build_prompt(adsb_data, transmissions)

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,
                        "num_predict": 4096,
                        "top_p": 0.9,
                    },
                },
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            result = response.json()
            return self._parse_response(result.get("response", ""))
        except requests.exceptions.Timeout:
            return {"error": "LLM request timed out", "raw": ""}
        except requests.exceptions.ConnectionError:
            return {"error": "Cannot connect to Ollama. Is it running?", "raw": ""}
        except Exception as exc:  # pragma: no cover - defensive
            return {"error": str(exc), "raw": ""}

    def _parse_response(self, text: str) -> dict:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                json_str = text[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        return {"error": "Failed to parse LLM response", "raw": text}


# ---------------------------------------------------------------------------
# Helper functions used by the monitoring stack
# ---------------------------------------------------------------------------

def build_adsb_contacts(aircraft_iterable: Iterable[Aircraft]) -> List[ADSBContact]:
    """Convert an iterable of :class:`Aircraft` objects into contacts."""
    contacts: List[ADSBContact] = []
    for aircraft in aircraft_iterable:
        if not aircraft:
            continue
        try:
            contact = ADSBContact.from_aircraft(aircraft)
            contacts.append(contact)
        except Exception:
            continue
    return contacts


def _compute_audio_duration(segments: Sequence[dict]) -> float:
    if not segments:
        return 0.0
    starts = [seg.get("start") for seg in segments if seg.get("start") is not None]
    ends = [seg.get("end") for seg in segments if seg.get("end") is not None]
    if not starts or not ends:
        return 0.0
    try:
        return float(max(ends) - min(starts))
    except Exception:
        return 0.0


def _parse_timestamp(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        try:
            return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp()
        except ValueError:
            return None


def build_atc_transmission(
    text: str,
    frequency: str,
    channel_name: str,
    segments: Sequence[dict],
    *,
    recorded_timestamp: Optional[float] = None,
    metadata_timestamp: Optional[str] = None,
) -> ATCTransmission:
    """Create an :class:`ATCTransmission` from transcription metadata."""
    transcription_time = _parse_timestamp(metadata_timestamp)
    if transcription_time is None:
        transcription_time = time.time()

    audio_duration = _compute_audio_duration(segments)

    transcription_delay = 0.0
    if recorded_timestamp is not None:
        transcription_delay = max(0.0, transcription_time - recorded_timestamp)

    return ATCTransmission(
        timestamp=transcription_time,
        text=text,
        frequency=str(frequency),
        channel_name=channel_name,
        segments=list(segments),
        audio_duration=audio_duration,
        transcription_delay=transcription_delay,
    )


__all__ = [
    "ADSBContact",
    "ATCTransmission",
    "ContextBuilder",
    "OllamaCorrelator",
    "build_adsb_contacts",
    "build_atc_transmission",
]
