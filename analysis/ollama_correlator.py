"""Integration helpers for performing ADS-B/ATC correlation via Ollama LLM."""
from __future__ import annotations

import json
import math
import time
import threading
import queue
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Dict, Any

import requests

try:
    import tkinter as tk
    from tkinter import scrolledtext, ttk
    HAS_TK = True
except ImportError:
    HAS_TK = False

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
        # Some data sources (including the default OpenSky implementation) expose
        # the aircraft's course as ``track`` rather than ``heading``.  The
        # previous implementation attempted to access ``aircraft.heading``
        # directly which raises ``AttributeError`` for those objects.  Because
        # ``build_adsb_contacts`` swallows any exception when converting an
        # aircraft to a contact, the AttributeError caused every aircraft to be
        # dropped and resulted in the LLM correlator receiving an empty ADS-B
        # dataset.  Using ``getattr`` preserves compatibility with sources that
        # provide either attribute without raising.
        heading = getattr(aircraft, "heading", None)

        return cls(
            icao=(aircraft.icao24 or "").upper(),
            callsign=callsign,
            altitude=int(aircraft.altitude or 0),
            heading=int(heading or aircraft.track or 0),
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


class DebugMonitor:
    """Debug window for monitoring LLM interactions."""

    def __init__(self):
        self.message_queue: queue.Queue = queue.Queue()
        self.window = None
        self.text_widget = None
        self.running = False
        self._thread = None

        if HAS_TK:
            self._start_gui_thread()

    def _start_gui_thread(self):
        """Start the GUI in a separate thread."""
        self._thread = threading.Thread(target=self._run_gui, daemon=True)
        self._thread.start()

    def _run_gui(self):
        """Run the tkinter GUI."""
        self.window = tk.Tk()
        self.window.title("LLM Correlator Debug Monitor")
        self.window.geometry("1200x800")

        # Create main frame
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Stats frame
        stats_frame = ttk.LabelFrame(main_frame, text="Statistics")
        stats_frame.pack(fill=tk.X, pady=(0, 5))

        self.stats_labels = {}
        stats = ["API Calls", "Total Tokens", "Avg Response Time", "Errors", "Context Size"]
        for i, stat in enumerate(stats):
            ttk.Label(stats_frame, text=f"{stat}:").grid(row=0, column=i*2, padx=5, pady=2)
            self.stats_labels[stat] = ttk.Label(stats_frame, text="0")
            self.stats_labels[stat].grid(row=0, column=i*2+1, padx=5, pady=2)

        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Log tab
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="Activity Log")

        self.text_widget = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, font=("Consolas", 9)
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True)

        # Configure tags for coloring
        self.text_widget.tag_configure("timestamp", foreground="gray")
        self.text_widget.tag_configure("prompt", foreground="blue")
        self.text_widget.tag_configure("response", foreground="green")
        self.text_widget.tag_configure("error", foreground="red")
        self.text_widget.tag_configure("info", foreground="black")
        self.text_widget.tag_configure("warning", foreground="orange")

        # Prompts tab
        prompt_frame = ttk.Frame(notebook)
        notebook.add(prompt_frame, text="Last Prompt")

        self.prompt_text = scrolledtext.ScrolledText(
            prompt_frame, wrap=tk.WORD, font=("Consolas", 9)
        )
        self.prompt_text.pack(fill=tk.BOTH, expand=True)

        # Response tab
        response_frame = ttk.Frame(notebook)
        notebook.add(response_frame, text="Last Response")

        self.response_text = scrolledtext.ScrolledText(
            response_frame, wrap=tk.WORD, font=("Consolas", 9)
        )
        self.response_text.pack(fill=tk.BOTH, expand=True)

        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(button_frame, text="Clear Log", command=self._clear_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Log", command=self._save_log).pack(side=tk.LEFT, padx=5)

        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(button_frame, text="Auto-scroll", variable=self.auto_scroll_var).pack(side=tk.LEFT, padx=5)

        self.running = True
        self._process_queue()
        self.window.mainloop()

    def _process_queue(self):
        """Process messages from the queue."""
        try:
            while True:
                msg = self.message_queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass

        if self.running and self.window:
            self.window.after(100, self._process_queue)

    def _handle_message(self, msg: Dict[str, Any]):
        """Handle a message from the queue."""
        msg_type = msg.get("type", "info")

        if msg_type == "log":
            self._append_log(msg.get("text", ""), msg.get("tag", "info"))
        elif msg_type == "prompt":
            self.prompt_text.delete(1.0, tk.END)
            self.prompt_text.insert(tk.END, msg.get("text", ""))
            self._append_log(f"Sent prompt ({len(msg.get('text', ''))} chars)", "prompt")
        elif msg_type == "response":
            self.response_text.delete(1.0, tk.END)
            self.response_text.insert(tk.END, msg.get("text", ""))
            self._append_log(f"Received response ({len(msg.get('text', ''))} chars)", "response")
        elif msg_type == "stats":
            for key, value in msg.get("data", {}).items():
                if key in self.stats_labels:
                    self.stats_labels[key].config(text=str(value))

    def _append_log(self, text: str, tag: str = "info"):
        """Append text to the log."""
        if self.text_widget:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.text_widget.insert(tk.END, f"[{timestamp}] ", "timestamp")
            self.text_widget.insert(tk.END, f"{text}\n", tag)

            if self.auto_scroll_var.get():
                self.text_widget.see(tk.END)

    def _clear_log(self):
        """Clear the log."""
        if self.text_widget:
            self.text_widget.delete(1.0, tk.END)

    def _save_log(self):
        """Save the log to a file."""
        if self.text_widget:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"correlator_log_{timestamp}.txt"
            with open(filename, "w") as f:
                f.write(self.text_widget.get(1.0, tk.END))
            self._append_log(f"Log saved to {filename}", "info")

    def log(self, text: str, tag: str = "info"):
        """Log a message."""
        self.message_queue.put({"type": "log", "text": text, "tag": tag})

    def log_prompt(self, text: str):
        """Log a prompt."""
        self.message_queue.put({"type": "prompt", "text": text})

    def log_response(self, text: str):
        """Log a response."""
        self.message_queue.put({"type": "response", "text": text})

    def update_stats(self, stats: Dict[str, Any]):
        """Update statistics."""
        self.message_queue.put({"type": "stats", "data": stats})


class RollingContextManager:
    """Manages rolling context window for the LLM."""

    def __init__(self, max_context_tokens: int = 8192, reserve_tokens: int = 1500):
        self.max_context_tokens = max_context_tokens
        self.reserve_tokens = reserve_tokens  # Reserve for response
        self.available_tokens = max_context_tokens - reserve_tokens

        # Approximate token counts (conservative estimates)
        self.chars_per_token = 3.5  # Average for mixed text

        self.system_prompt: str = ""
        self.system_prompt_tokens: int = 0

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt."""
        self.system_prompt = prompt
        self.system_prompt_tokens = self._estimate_tokens(prompt)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return int(len(text) / self.chars_per_token) + 10  # Add buffer

    def build_context_prompt(
        self,
        adsb_data: Sequence[ADSBContact],
        transmissions: Sequence[ATCTransmission],
        format_adsb_func,
        format_transmissions_func,
    ) -> tuple[str, int, int]:
        """Build prompt that fits within context window.

        Returns:
            Tuple of (prompt, num_adsb_included, num_tx_included)
        """
        available = self.available_tokens - self.system_prompt_tokens

        # Start with analysis template
        template = self._get_analysis_template()
        template_tokens = self._estimate_tokens(template)
        available -= template_tokens

        # Allocate space: 60% for ADS-B, 40% for transmissions
        adsb_budget = int(available * 0.6)
        tx_budget = available - adsb_budget

        # Select ADS-B contacts that fit
        adsb_included = []
        adsb_tokens = 0
        for contact in reversed(adsb_data):  # Most recent first
            contact_str = self._format_single_adsb(contact, format_adsb_func)
            contact_tokens = self._estimate_tokens(contact_str)
            if adsb_tokens + contact_tokens <= adsb_budget:
                adsb_included.insert(0, contact)
                adsb_tokens += contact_tokens
            else:
                break

        # Select transmissions that fit
        tx_included = []
        tx_tokens = 0
        for tx in reversed(transmissions):  # Most recent first
            tx_str = self._format_single_tx(tx, len(tx_included), format_transmissions_func)
            tx_tokens_est = self._estimate_tokens(tx_str)
            if tx_tokens + tx_tokens_est <= tx_budget:
                tx_included.insert(0, tx)
                tx_tokens += tx_tokens_est
            else:
                break

        # Build final prompt
        adsb_text = format_adsb_func(adsb_included)
        tx_text = format_transmissions_func(tx_included)

        prompt = f"{self.system_prompt}\n\n{template.format(adsb_data=adsb_text, tx_data=tx_text)}"

        return prompt, len(adsb_included), len(tx_included)

    def _format_single_adsb(self, contact: ADSBContact, format_func) -> str:
        """Format a single contact for size estimation."""
        return format_func([contact])

    def _format_single_tx(self, tx: ATCTransmission, idx: int, format_func) -> str:
        """Format a single transmission for size estimation."""
        return format_func([tx])

    def _get_analysis_template(self) -> str:
        """Get the analysis prompt template."""
        return """CURRENT ADS-B CONTACTS:
{adsb_data}

RECENT ATC TRANSMISSIONS TO ANALYZE:
{tx_data}

Analyze each transmission and respond with JSON matching the specified format.
Remember: Match callsigns flexibly (DAL2617="delta 26 17", UAL="united", AAL="american", SWA="southwest").
Only alert for NON_TRANSPONDER if callsign is CLEARLY extracted AND confirmed absent from ADS-B data."""


class ContextBuilder:
    """Builds the prompt used to query the LLM."""

    def __init__(self, max_adsb_contacts: int = 100, max_transmissions: int = 25):
        self.max_adsb = max_adsb_contacts
        self.max_tx = max_transmissions

    def build_system_prompt(self) -> str:
        """Build the system prompt that establishes context and rules."""
        return """You are an aviation ATC correlation analyst matching radio transmissions to ADS-B data.

CRITICAL: FLEXIBLE CALLSIGN MATCHING
Airlines use ICAO codes in ADS-B but phonetic names on radio:
- DAL/DL = "Delta" (e.g., DAL2617 = "Delta 26 17" or "Delta 2617")
- UAL/UA = "United"
- AAL/AA = "American"  
- SWA/WN = "Southwest"
- J BU/B6 = "JetBlue"
- SKW = "SkyWest"
- ENY = "Envoy"
- RPA = "Republic"
- ASA/AS = "Alaska"
- FFT = "Frontier"
- NKS = "Spirit"
- VIR = "Virgin"

NUMBER MATCHING - BE FLEXIBLE:
- "Delta 26 17" = DAL2617 ✓
- "Delta twenty-six seventeen" = DAL2617 ✓
- "Delta 2 6 1 7" = DAL2617 ✓
- Numbers might be spoken with pauses or grouped differently

GENERAL AVIATION:
- N-numbers: "November 1 2 3 Alpha Bravo" = N123AB
- Cessna/Piper/etc followed by tail number

TRANSCRIPTION QUALITY:
- Expect errors: "data" might be "delta", numbers may be wrong
- Use context clues: altitude, location mentioned
- Partial matches are valuable - note them

ALERTING RULES:
- Match aircraft when reasonably confident (>60%)
- Flag NON_TRANSPONDER only when:
  1. Callsign clearly extracted (not garbled)
  2. Definitely not in ADS-B list (check carefully!)
  3. Confidence > 70%
- Flag MILITARY for: REACH/RCH, VIPER, EAGLE, HAMMER, KING, RESCUE, EVAC, DUKE

OUTPUT FORMAT:
{
  "correlations": [
    {
      "transmission_id": <index>,
      "extracted_identifier": "<what you heard>",
      "extraction_confidence": <0.0-1.0>,
      "matched_icao": "<ICAO or NO_MATCH or UNCLEAR>",
      "matched_callsign": "<ADS-B callsign if matched>",
      "match_confidence": <0.0-1.0>,
      "reasoning": "<explanation including airline code matching>",
      "flags": []
    }
  ],
  "alerts": [
    {
      "type": "<MILITARY|NON_TRANSPONDER>",
      "callsign": "<extracted>",
      "details": "<why alerting>",
      "severity": "<HIGH|MEDIUM|LOW>",
      "confidence": <must be >0.7>
    }
  ],
  "summary": "<brief assessment>"
}"""

    def build_analysis_prompt(
        self,
        adsb_data: Sequence[ADSBContact],
        transmissions: Sequence[ATCTransmission],
    ) -> str:
        """Build the analysis prompt with current data."""
        adsb_summary = self._format_adsb(adsb_data[-self.max_adsb :])
        tx_summary = self._format_transmissions(transmissions[-self.max_tx :])

        return f"""CURRENT ADS-B CONTACTS ({len(adsb_data)} tracked):
{adsb_summary}

RECENT ATC TRANSMISSIONS:
{tx_summary}

Analyze each transmission. Match callsigns FLEXIBLY (DAL2617 = "delta 26 17").
Respond with JSON only."""

    def _format_adsb(self, contacts: Sequence[ADSBContact]) -> str:
        lines: List[str] = []
        for contact in contacts:
            callsign = contact.callsign or "--------"
            squawk = contact.squawk or "----"
            age = time.time() - contact.timestamp
            age_str = f"{int(age)}s" if age < 120 else f"{int(age/60)}m"
            lines.append(
                f"{contact.icao} {callsign:8} {contact.altitude:5}ft "
                f"{contact.heading:03}° {contact.speed:3}kt {squawk} ({age_str})"
            )
        return "\n".join(lines) if lines else "(no contacts)"

    def _format_transmissions(self, txs: Sequence[ATCTransmission]) -> str:
        lines: List[str] = []
        for idx, tx in enumerate(txs):
            age = time.time() - tx.timestamp
            age_str = f"{int(age)}s"
            # Truncate very long transmissions
            text = tx.text[:200] + "..." if len(tx.text) > 200 else tx.text
            lines.append(
                f"[{idx}] ({age_str}) {tx.channel_name}: \"{text}\""
            )
        return "\n".join(lines) if lines else "(no transmissions)"


class OllamaCorrelator:
    """Query an Ollama-hosted model to correlate ATC transmissions."""

    def __init__(
        self,
        model: str = "gpt-oss:20b",
        base_url: str = "http://localhost:11434",
        max_adsb_contacts: int = 100,
        max_transmissions: int = 25,
        request_timeout: int = 120,
        enable_debug_monitor: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.context_builder = ContextBuilder(
            max_adsb_contacts=max_adsb_contacts,
            max_transmissions=max_transmissions,
        )
        self.rolling_context = RollingContextManager(
            max_context_tokens=8192,
            reserve_tokens=2000  # Reserve space for response
        )
        self.request_timeout = request_timeout

        # Statistics
        self.stats = {
            "API Calls": 0,
            "Total Tokens": 0,
            "Avg Response Time": "0.0s",
            "Errors": 0,
            "Context Size": "0/8192",
        }
        self._response_times: List[float] = []

        # Debug monitor
        self.debug_monitor = None
        if enable_debug_monitor and HAS_TK:
            self.debug_monitor = DebugMonitor()
            time.sleep(0.5)  # Give GUI time to initialize

        # Initialize system prompt
        system_prompt = self.context_builder.build_system_prompt()
        self.rolling_context.set_system_prompt(system_prompt)

        self._log("Correlator initialized", "info")
        self._log(f"Model: {model}", "info")
        self._log(f"System prompt: {self.rolling_context.system_prompt_tokens} tokens (est)", "info")

    def _log(self, message: str, tag: str = "info"):
        """Log a message to the debug monitor."""
        if self.debug_monitor:
            self.debug_monitor.log(message, tag)
        else:
            print(f"[{tag.upper()}] {message}")

    def _update_stats(self):
        """Update statistics display."""
        if self.debug_monitor:
            self.debug_monitor.update_stats(self.stats)

    def correlate(
        self,
        adsb_data: Sequence[ADSBContact],
        transmissions: Sequence[ATCTransmission],
    ) -> dict:
        """Correlate transmissions with ADS-B data."""

        if not transmissions:
            self._log("No transmissions to analyze", "warning")
            return {"correlations": [], "alerts": [], "summary": "No transmissions"}

        start_time = time.time()
        self.stats["API Calls"] += 1

        # Build prompt with rolling context management
        prompt, num_adsb, num_tx = self.rolling_context.build_context_prompt(
            adsb_data,
            transmissions,
            self.context_builder._format_adsb,
            self.context_builder._format_transmissions,
        )

        prompt_tokens = self.rolling_context._estimate_tokens(prompt)
        self.stats["Context Size"] = f"{prompt_tokens}/8192"
        self.stats["Total Tokens"] += prompt_tokens

        self._log(f"Built prompt: {num_adsb} contacts, {num_tx} transmissions, ~{prompt_tokens} tokens", "info")

        if self.debug_monitor:
            self.debug_monitor.log_prompt(prompt)

        try:
            self._log(f"Sending request to {self.model}...", "info")

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.4,
                        "num_predict": 2000,
                        "top_p": 0.9,
                        "num_ctx": 8192,
                        "stop": ["\n\n\n"],  # Stop on triple newline
                    },
                },
                timeout=self.request_timeout,
            )
            response.raise_for_status()

            elapsed = time.time() - start_time
            self._response_times.append(elapsed)
            if len(self._response_times) > 100:
                self._response_times.pop(0)
            avg_time = sum(self._response_times) / len(self._response_times)
            self.stats["Avg Response Time"] = f"{avg_time:.1f}s"

            result = response.json()
            response_text = result.get("response", "")

            if self.debug_monitor:
                self.debug_monitor.log_response(response_text)

            # Update token stats
            eval_count = result.get("eval_count", 0)
            self.stats["Total Tokens"] += eval_count

            self._log(f"Response received in {elapsed:.1f}s ({eval_count} tokens)", "response")

            parsed = self._parse_response(response_text)

            # Post-process alerts
            if "alerts" in parsed and isinstance(parsed["alerts"], list):
                original_count = len(parsed["alerts"])
                parsed["alerts"] = [
                    alert for alert in parsed["alerts"]
                    if alert.get("confidence", 0) >= 0.7
                ]
                filtered = original_count - len(parsed["alerts"])
                if filtered > 0:
                    self._log(f"Filtered {filtered} low-confidence alerts", "info")

            # Log correlations
            if "correlations" in parsed:
                matches = sum(1 for c in parsed["correlations"]
                            if c.get("matched_icao") not in ["NO_MATCH", "UNCLEAR", None])
                self._log(f"Found {matches}/{len(parsed['correlations'])} matches", "info")

            # Log alerts
            if parsed.get("alerts"):
                for alert in parsed["alerts"]:
                    self._log(
                        f"ALERT: {alert.get('type')} - {alert.get('callsign')} "
                        f"(conf: {alert.get('confidence', 0):.2f})",
                        "warning"
                    )

            self._update_stats()
            return parsed

        except requests.exceptions.Timeout:
            self.stats["Errors"] += 1
            self._log("Request timed out", "error")
            self._update_stats()
            return {"error": "LLM request timed out", "raw": ""}
        except requests.exceptions.ConnectionError:
            self.stats["Errors"] += 1
            self._log("Cannot connect to Ollama", "error")
            self._update_stats()
            return {"error": "Cannot connect to Ollama", "raw": ""}
        except Exception as exc:
            self.stats["Errors"] += 1
            self._log(f"Error: {exc}", "error")
            self._update_stats()
            return {"error": str(exc), "raw": ""}

    def _parse_response(self, text: str) -> dict:
        """Parse JSON from LLM response."""
        try:
            # Find JSON in response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                json_str = text[start:end]
                parsed = json.loads(json_str)

                # Validate and ensure structure
                if "correlations" not in parsed:
                    parsed["correlations"] = []
                if "alerts" not in parsed:
                    parsed["alerts"] = []
                if "summary" not in parsed:
                    parsed["summary"] = "Analysis complete"

                self._log("Successfully parsed JSON response", "info")
                return parsed
        except json.JSONDecodeError as e:
            self._log(f"JSON parse error: {e}", "error")

        self._log("Failed to parse response as JSON", "error")
        return {"error": "Failed to parse LLM response", "raw": text}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_adsb_contacts(aircraft_iterable: Iterable[Aircraft]) -> List[ADSBContact]:
    """Convert an iterable of Aircraft objects into contacts."""
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
    """Create an ATCTransmission from transcription metadata."""
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
    "DebugMonitor",
    "RollingContextManager",
    "build_adsb_contacts",
    "build_atc_transmission",
]