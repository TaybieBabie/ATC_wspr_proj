"""Integration helpers for performing ADS-B/ATC correlation via Ollama LLM."""
from __future__ import annotations

import json
import math
import time
import threading
import queue
from dataclasses import dataclass
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
    """Debug window for monitoring LLM interactions with dark terminal theme."""

    COLORS = {
        'bg': '#0d1117',
        'fg': '#c9d1d9',
        'bg_secondary': '#161b22',
        'bg_tertiary': '#21262d',
        'border': '#30363d',
        'accent': '#58a6ff',
        'timestamp': '#8b949e',
        'prompt': '#79c0ff',
        'response': '#7ee787',
        'error': '#f85149',
        'warning': '#d29922',
        'info': '#c9d1d9',
        'success': '#3fb950',
    }

    def __init__(self):
        self.message_queue: queue.Queue = queue.Queue()
        self.window = None
        self.text_widget = None
        self.running = False
        self._thread = None

        if HAS_TK:
            self._start_gui_thread()

    def _start_gui_thread(self):
        self._thread = threading.Thread(target=self._run_gui, daemon=True)
        self._thread.start()

    def _configure_dark_theme(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('Dark.TFrame', background=self.COLORS['bg'])
        style.configure('DarkSecondary.TFrame', background=self.COLORS['bg_secondary'])
        style.configure('Dark.TLabelframe', background=self.COLORS['bg_secondary'],
                       foreground=self.COLORS['fg'], bordercolor=self.COLORS['border'])
        style.configure('Dark.TLabelframe.Label', background=self.COLORS['bg_secondary'],
                       foreground=self.COLORS['accent'], font=('Consolas', 10, 'bold'))
        style.configure('Dark.TLabel', background=self.COLORS['bg_secondary'],
                       foreground=self.COLORS['fg'], font=('Consolas', 9))
        style.configure('DarkValue.TLabel', background=self.COLORS['bg_secondary'],
                       foreground=self.COLORS['success'], font=('Consolas', 9, 'bold'))
        style.configure('Dark.TNotebook', background=self.COLORS['bg'],
                       bordercolor=self.COLORS['border'])
        style.configure('Dark.TNotebook.Tab', background=self.COLORS['bg_tertiary'],
                       foreground=self.COLORS['fg'], padding=[10, 5], font=('Consolas', 9))
        style.map('Dark.TNotebook.Tab',
                 background=[('selected', self.COLORS['bg_secondary'])],
                 foreground=[('selected', self.COLORS['accent'])])
        style.configure('Dark.TButton', background=self.COLORS['bg_tertiary'],
                       foreground=self.COLORS['fg'], bordercolor=self.COLORS['border'],
                       font=('Consolas', 9))
        style.map('Dark.TButton', background=[('active', self.COLORS['border'])],
                 foreground=[('active', self.COLORS['accent'])])
        style.configure('Dark.TCheckbutton', background=self.COLORS['bg'],
                       foreground=self.COLORS['fg'], font=('Consolas', 9))

    def _run_gui(self):
        self.window = tk.Tk()
        self.window.title("LLM Correlator Debug Monitor")
        self.window.geometry("1200x800")
        self.window.configure(bg=self.COLORS['bg'])
        self._configure_dark_theme()

        main_frame = ttk.Frame(self.window, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        stats_frame = ttk.LabelFrame(main_frame, text="Statistics", style='Dark.TLabelframe')
        stats_frame.pack(fill=tk.X, pady=(0, 5))

        self.stats_labels = {}
        stats = ["API Calls", "Total Tokens", "Avg Response Time", "Errors", "Context Size"]
        for i, stat in enumerate(stats):
            ttk.Label(stats_frame, text=f"{stat}:", style='Dark.TLabel').grid(
                row=0, column=i*2, padx=5, pady=5)
            self.stats_labels[stat] = ttk.Label(stats_frame, text="0", style='DarkValue.TLabel')
            self.stats_labels[stat].grid(row=0, column=i*2+1, padx=5, pady=5)

        notebook = ttk.Notebook(main_frame, style='Dark.TNotebook')
        notebook.pack(fill=tk.BOTH, expand=True)

        log_frame = ttk.Frame(notebook, style='DarkSecondary.TFrame')
        notebook.add(log_frame, text="Activity Log")

        self.text_widget = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, font=("Consolas", 10),
            bg=self.COLORS['bg'], fg=self.COLORS['fg'],
            insertbackground=self.COLORS['accent'],
            selectbackground=self.COLORS['border'],
            selectforeground=self.COLORS['fg'],
            relief=tk.FLAT, borderwidth=0, padx=10, pady=10
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        for tag, color in [("timestamp", 'timestamp'), ("prompt", 'prompt'),
                          ("response", 'response'), ("error", 'error'),
                          ("info", 'info'), ("warning", 'warning'), ("success", 'success')]:
            self.text_widget.tag_configure(tag, foreground=self.COLORS[color])

        prompt_frame = ttk.Frame(notebook, style='DarkSecondary.TFrame')
        notebook.add(prompt_frame, text="Last Prompt")
        self.prompt_text = scrolledtext.ScrolledText(
            prompt_frame, wrap=tk.WORD, font=("Consolas", 10),
            bg=self.COLORS['bg'], fg=self.COLORS['prompt'],
            insertbackground=self.COLORS['accent'],
            selectbackground=self.COLORS['border'],
            selectforeground=self.COLORS['fg'],
            relief=tk.FLAT, borderwidth=0, padx=10, pady=10
        )
        self.prompt_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        response_frame = ttk.Frame(notebook, style='DarkSecondary.TFrame')
        notebook.add(response_frame, text="Last Response")
        self.response_text = scrolledtext.ScrolledText(
            response_frame, wrap=tk.WORD, font=("Consolas", 10),
            bg=self.COLORS['bg'], fg=self.COLORS['response'],
            insertbackground=self.COLORS['accent'],
            selectbackground=self.COLORS['border'],
            selectforeground=self.COLORS['fg'],
            relief=tk.FLAT, borderwidth=0, padx=10, pady=10
        )
        self.response_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        button_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        button_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(button_frame, text="Clear Log", command=self._clear_log,
                  style='Dark.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Log", command=self._save_log,
                  style='Dark.TButton').pack(side=tk.LEFT, padx=5)

        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(button_frame, text="Auto-scroll", variable=self.auto_scroll_var,
                       style='Dark.TCheckbutton').pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(button_frame, text="● Ready", style='DarkValue.TLabel')
        self.status_label.pack(side=tk.RIGHT, padx=10)

        self.running = True
        self._process_queue()
        self.window.mainloop()

    def _process_queue(self):
        try:
            while True:
                msg = self.message_queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        if self.running and self.window:
            self.window.after(100, self._process_queue)

    def _handle_message(self, msg: Dict[str, Any]):
        msg_type = msg.get("type", "info")
        if msg_type == "log":
            self._append_log(msg.get("text", ""), msg.get("tag", "info"))
        elif msg_type == "prompt":
            self.prompt_text.delete(1.0, tk.END)
            self.prompt_text.insert(tk.END, msg.get("text", ""))
            self._append_log(f"▶ Sent prompt ({len(msg.get('text', ''))} chars)", "prompt")
        elif msg_type == "response":
            self.response_text.delete(1.0, tk.END)
            self.response_text.insert(tk.END, msg.get("text", ""))
            self._append_log(f"◀ Received response ({len(msg.get('text', ''))} chars)", "response")
        elif msg_type == "stats":
            for key, value in msg.get("data", {}).items():
                if key in self.stats_labels:
                    self.stats_labels[key].config(text=str(value))
        elif msg_type == "status":
            if hasattr(self, 'status_label'):
                self.status_label.config(text=msg.get("text", ""))

    def _append_log(self, text: str, tag: str = "info"):
        if self.text_widget:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.text_widget.insert(tk.END, f"[{timestamp}] ", "timestamp")
            self.text_widget.insert(tk.END, f"{text}\n", tag)
            if self.auto_scroll_var.get():
                self.text_widget.see(tk.END)

    def _clear_log(self):
        if self.text_widget:
            self.text_widget.delete(1.0, tk.END)
            self._append_log("Log cleared", "info")

    def _save_log(self):
        if self.text_widget:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"correlator_log_{timestamp}.txt"
            with open(filename, "w") as f:
                f.write(self.text_widget.get(1.0, tk.END))
            self._append_log(f"Log saved to {filename}", "success")

    def log(self, text: str, tag: str = "info"):
        self.message_queue.put({"type": "log", "text": text, "tag": tag})

    def log_prompt(self, text: str):
        self.message_queue.put({"type": "prompt", "text": text})

    def log_response(self, text: str):
        self.message_queue.put({"type": "response", "text": text})

    def update_stats(self, stats: Dict[str, Any]):
        self.message_queue.put({"type": "stats", "data": stats})

    def set_status(self, text: str):
        self.message_queue.put({"type": "status", "text": text})


class RollingContextManager:
    """Manages rolling context window for the LLM with strict budget enforcement."""

    def __init__(self, max_context_tokens: int = 8192, max_response_tokens: int = 2048):
        self.max_context_tokens = max_context_tokens
        self.max_response_tokens = max_response_tokens
        # Total available = context - response reserve
        self.max_prompt_tokens = max_context_tokens - max_response_tokens

        # More conservative token estimation for JSON responses
        self.chars_per_token = 4.0  # Increased from 3.5 for safety
        self.tokens_per_correlation = 180  # Estimated tokens per correlation entry

        self.system_prompt: str = ""
        self.system_prompt_tokens: int = 0

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt
        self.system_prompt_tokens = self._estimate_tokens(prompt)

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text) / self.chars_per_token) + 20  # Larger buffer

    def calculate_max_transmissions(self, num_adsb: int) -> int:
        """Calculate how many transmissions we can safely process."""
        # Each transmission needs ~180 tokens in response
        available_response = self.max_response_tokens - 200  # Reserve for JSON wrapper
        max_tx = available_response // self.tokens_per_correlation
        return min(max_tx, 10)  # Hard cap at 10 transmissions

    def build_context_prompt(
        self,
        adsb_data: Sequence[ADSBContact],
        transmissions: Sequence[ATCTransmission],
        format_adsb_func,
        format_transmissions_func,
    ) -> tuple[str, int, int]:
        """Build prompt that fits within context window."""
        # Calculate available prompt budget
        available = self.max_prompt_tokens - self.system_prompt_tokens

        template = self._get_analysis_template()
        template_tokens = self._estimate_tokens(template)
        available -= template_tokens

        # 70% for ADS-B, 30% for transmissions (ADS-B is more compact)
        adsb_budget = int(available * 0.70)
        tx_budget = available - adsb_budget

        # Select ADS-B contacts that fit
        adsb_included = []
        adsb_tokens = 0
        for contact in reversed(adsb_data):
            contact_str = format_adsb_func([contact])
            contact_tokens = self._estimate_tokens(contact_str)
            if adsb_tokens + contact_tokens <= adsb_budget:
                adsb_included.insert(0, contact)
                adsb_tokens += contact_tokens
            else:
                break

        # Calculate max transmissions based on response budget
        max_tx = self.calculate_max_transmissions(len(adsb_included))

        # Select transmissions that fit both prompt budget AND response budget
        tx_included = []
        tx_tokens = 0
        for tx in reversed(transmissions):
            if len(tx_included) >= max_tx:
                break
            tx_str = format_transmissions_func([tx])
            tx_tokens_est = self._estimate_tokens(tx_str)
            if tx_tokens + tx_tokens_est <= tx_budget:
                tx_included.insert(0, tx)
                tx_tokens += tx_tokens_est
            else:
                break

        adsb_text = format_adsb_func(adsb_included)
        tx_text = format_transmissions_func(tx_included)

        prompt = f"{self.system_prompt}\n\n{template.format(adsb_data=adsb_text, tx_data=tx_text, num_tx=len(tx_included))}"

        return prompt, len(adsb_included), len(tx_included)

    def _get_analysis_template(self) -> str:
        return """ADS-B CONTACTS:
{adsb_data}

TRANSMISSIONS ({num_tx} total):
{tx_data}

Respond with COMPACT JSON. Keep reasoning brief (max 50 chars).
Ensure complete, valid JSON with all brackets closed."""


class ContextBuilder:
    """Builds the prompt used to query the LLM."""

    def __init__(self, max_adsb_contacts: int = 100, max_transmissions: int = 10):
        self.max_adsb = max_adsb_contacts
        self.max_tx = max_transmissions

    def build_system_prompt(self) -> str:
        return """ATC correlation analyst. Match transmissions to ADS-B using fuzzy logic.

AIRLINE CODES: DAL=Delta UAL=United AAL=American SWA=Southwest JBU=JetBlue ASA=Alaska SKW=SkyWest ENY=Envoy RPA=Republic FFT=Frontier NKS=Spirit

FUZZY MATCH RULES:
- Numbers: DAL2617="Delta 26 17"="Delta 2617"="Delta 2 6 1 7"
- Phonetic: tree=3 fife=5 niner=9
- Errors: 5↔9 B↔D↔P M↔N common, off-by-one OK
- Partial: "Delta 617"→DAL2617/DAL1617, suffix "...17"→*17
- Transcription: "delta"→"data"/"dealt", split words OK

CONFIDENCE: exact=0.95, 1-digit-off=0.80, partial=0.70, fuzzy=0.60, unclear=0.30

MATCH if conf≥0.50. ALERT only if: extraction≥0.80, definitely absent, alert_conf≥0.70

MILITARY: REACH RCH VIPER EAGLE HAMMER KING RESCUE EVAC DUKE

OUTPUT (keep reasoning SHORT):
{"correlations":[{"transmission_id":0,"extracted_identifier":"text","extraction_confidence":0.8,"matched_icao":"ICAO/NO_MATCH/UNCLEAR","matched_callsign":"CS","match_confidence":0.7,"reasoning":"brief","flags":[]}],"alerts":[],"summary":"brief"}"""

    def _format_adsb(self, contacts: Sequence[ADSBContact]) -> str:
        lines: List[str] = []
        for contact in contacts:
            cs = contact.callsign or "----"
            lines.append(f"{contact.icao} {cs:8} {contact.altitude}ft {contact.heading:03}° {contact.speed}kt")
        return "\n".join(lines) if lines else "(none)"

    def _format_transmissions(self, txs: Sequence[ATCTransmission]) -> str:
        lines: List[str] = []
        for idx, tx in enumerate(txs):
            text = tx.text[:150] + "..." if len(tx.text) > 150 else tx.text
            lines.append(f"[{idx}] {tx.channel_name}: \"{text}\"")
        return "\n".join(lines) if lines else "(none)"


class OllamaCorrelator:
    """Query an Ollama-hosted model to correlate ATC transmissions."""

    def __init__(
        self,
        model: str = "gpt-oss:20b",
        base_url: str = "http://localhost:11434",
        max_adsb_contacts: int = 80,
        max_transmissions: int = 8,  # Reduced default
        request_timeout: int = 220,
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
            max_response_tokens=4096  # Strict response budget
        )
        self.request_timeout = request_timeout

        self.stats = {
            "API Calls": 0,
            "Total Tokens": 0,
            "Avg Response Time": "0.0s",
            "Errors": 0,
            "Context Size": "0/8192",
        }
        self._response_times: List[float] = []

        self.debug_monitor = None
        if enable_debug_monitor and HAS_TK:
            self.debug_monitor = DebugMonitor()
            time.sleep(0.5)

        system_prompt = self.context_builder.build_system_prompt()
        self.rolling_context.set_system_prompt(system_prompt)

        self._log("Correlator initialized", "info")
        self._log(f"Model: {model}", "info")
        self._log(f"System prompt: ~{self.rolling_context.system_prompt_tokens} tokens", "info")
        self._log(f"Max prompt: {self.rolling_context.max_prompt_tokens} tokens", "info")
        self._log(f"Max response: {self.rolling_context.max_response_tokens} tokens", "info")

    def _log(self, message: str, tag: str = "info"):
        if self.debug_monitor:
            self.debug_monitor.log(message, tag)
        else:
            print(f"[{tag.upper()}] {message}")

    def _update_stats(self):
        if self.debug_monitor:
            self.debug_monitor.update_stats(self.stats)

    def _set_status(self, status: str):
        if self.debug_monitor:
            self.debug_monitor.set_status(status)

    def correlate(
        self,
        adsb_data: Sequence[ADSBContact],
        transmissions: Sequence[ATCTransmission],
    ) -> dict:
        if not transmissions:
            self._log("No transmissions to analyze", "warning")
            return {"correlations": [], "alerts": [], "summary": "No transmissions"}

        start_time = time.time()
        self.stats["API Calls"] += 1
        self._set_status("● Processing...")

        # Build prompt with strict budget management
        prompt, num_adsb, num_tx = self.rolling_context.build_context_prompt(
            adsb_data,
            transmissions,
            self.context_builder._format_adsb,
            self.context_builder._format_transmissions,
        )

        prompt_tokens = self.rolling_context._estimate_tokens(prompt)
        total_budget = prompt_tokens + self.rolling_context.max_response_tokens
        self.stats["Context Size"] = f"{prompt_tokens}+{self.rolling_context.max_response_tokens}/{self.rolling_context.max_context_tokens}"

        if total_budget > self.rolling_context.max_context_tokens:
            self._log(f"WARNING: Budget overflow! {total_budget} > {self.rolling_context.max_context_tokens}", "error")

        self._log(f"Prompt: {num_adsb} contacts, {num_tx} transmissions, ~{prompt_tokens} tokens", "info")
        self._log(f"Expected response: ~{num_tx * self.rolling_context.tokens_per_correlation} tokens", "info")

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
                        "temperature": 0.3,  # Lower for more consistent output
                        "num_predict": self.rolling_context.max_response_tokens,
                        "top_p": 0.9,
                        "num_ctx": self.rolling_context.max_context_tokens,
                        "repeat_penalty": 1.1,
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

            eval_count = result.get("eval_count", 0)
            prompt_eval_count = result.get("prompt_eval_count", 0)
            self.stats["Total Tokens"] += eval_count + prompt_eval_count

            self._log(f"Response: {elapsed:.1f}s (prompt: {prompt_eval_count}, response: {eval_count} tokens)", "response")

            # Check for truncation
            if eval_count >= self.rolling_context.max_response_tokens - 50:
                self._log("WARNING: Response likely truncated (hit token limit)", "warning")
                response_text = self._attempt_json_repair(response_text)
            elif not response_text.rstrip().endswith("}"):
                self._log("Warning: Response may be incomplete", "warning")
                response_text = self._attempt_json_repair(response_text)

            parsed = self._parse_response(response_text)

            if "error" in parsed:
                self.stats["Errors"] += 1
                self._update_stats()
                self._set_status("● Parse Error")
                return parsed

            # Filter low-confidence alerts
            if "alerts" in parsed and isinstance(parsed["alerts"], list):
                original_count = len(parsed["alerts"])
                parsed["alerts"] = [
                    alert for alert in parsed["alerts"]
                    if alert.get("confidence", 0) >= 0.7
                ]
                filtered = original_count - len(parsed["alerts"])
                if filtered > 0:
                    self._log(f"Filtered {filtered} low-confidence alerts", "info")

            if "correlations" in parsed:
                matches = sum(1 for c in parsed["correlations"]
                            if c.get("matched_icao") not in ["NO_MATCH", "UNCLEAR", None])
                self._log(f"Matches: {matches}/{len(parsed['correlations'])}", "success")

            if parsed.get("alerts"):
                for alert in parsed["alerts"]:
                    self._log(f"ALERT: {alert.get('type')} - {alert.get('callsign')} ({alert.get('confidence', 0):.2f})", "warning")

            self._update_stats()
            self._set_status("● Ready")
            return parsed

        except requests.exceptions.Timeout:
            self.stats["Errors"] += 1
            self._log(f"Request timed out after {self.request_timeout}s", "error")
            self._update_stats()
            self._set_status("● Timeout")
            return {"error": "LLM request timed out", "raw": ""}
        except requests.exceptions.ConnectionError:
            self.stats["Errors"] += 1
            self._log("Cannot connect to Ollama", "error")
            self._update_stats()
            self._set_status("● Connection Error")
            return {"error": "Cannot connect to Ollama", "raw": ""}
        except Exception as exc:
            self.stats["Errors"] += 1
            self._log(f"Error: {exc}", "error")
            self._update_stats()
            self._set_status("● Error")
            return {"error": str(exc), "raw": ""}

    def _attempt_json_repair(self, text: str) -> str:
        text = text.rstrip()

        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')

        if open_braces > 0 or open_brackets > 0:
            self._log(f"Repairing JSON: missing {open_braces} braces, {open_brackets} brackets", "warning")

            # Try to find last complete JSON element
            # Look for patterns that indicate end of valid data
            repair_point = len(text)

            # Find last complete correlation or alert entry
            for pattern in ['"}', "']", "},", "],", "}"]:
                pos = text.rfind(pattern)
                if pos > 0:
                    # Check if this is inside a string
                    repair_point = min(repair_point, pos + len(pattern))

            # Trim to last good position if we found partial content
            if repair_point < len(text):
                text = text[:repair_point]

            # Close all open brackets
            text += "]" * open_brackets + "}" * open_braces

            self._log("JSON repair applied", "info")

        return text

    def _parse_response(self, text: str) -> dict:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                json_str = text[start:end]
                parsed = json.loads(json_str)

                if "correlations" not in parsed:
                    parsed["correlations"] = []
                if "alerts" not in parsed:
                    parsed["alerts"] = []
                if "summary" not in parsed:
                    parsed["summary"] = "Analysis complete"

                self._log("JSON parsed successfully", "success")
                return parsed
        except json.JSONDecodeError as e:
            self._log(f"JSON parse error at pos {e.pos}: {e.msg}", "error")
            if hasattr(e, 'pos') and e.pos:
                start_ctx = max(0, e.pos - 30)
                end_ctx = min(len(text), e.pos + 30)
                self._log(f"Error context: ...{text[start_ctx:end_ctx]}...", "error")

        return {"error": "Failed to parse LLM response", "raw": text}


def build_adsb_contacts(aircraft_iterable: Iterable[Aircraft]) -> List[ADSBContact]:
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