"""Integration helpers for performing ADS-B/ATC correlation via Ollama LLM."""
from __future__ import annotations

import atexit
import gc
import json
import math
import os
import sys
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
from utils.config import (
    LLM_MAX_ADSB_CONTACTS,
    LLM_MAX_TRANSMISSIONS,
    OLLAMA_ADSB_PROMPT_RATIO,
    OLLAMA_ALERT_CONFIDENCE_THRESHOLD,
    OLLAMA_BASE_URL,
    OLLAMA_CHARS_PER_TOKEN,
    OLLAMA_CONTEXT_WINDOW_TOKENS,
    OLLAMA_DEBUG_MONITOR_DELAY,
    OLLAMA_ENABLE_DEBUG_MONITOR,
    OLLAMA_MAX_RESPONSE_TOKENS,
    OLLAMA_MAX_TRANSMISSION_BATCH,
    OLLAMA_MODEL,
    OLLAMA_REPEAT_PENALTY,
    OLLAMA_REQUEST_TIMEOUT,
    OLLAMA_RESPONSE_JSON_OVERHEAD,
    OLLAMA_RESPONSE_SAFETY_MARGIN,
    OLLAMA_RESPONSE_TIME_WINDOW,
    OLLAMA_TEMPERATURE,
    OLLAMA_TOKENS_PER_CORRELATION,
    OLLAMA_TOKEN_ESTIMATE_BUFFER,
    OLLAMA_TOP_P,
    OLLAMA_TRANSMISSION_PREVIEW_CHARS,
)

# ---------------------------------------------------------------------------
# Default keep_alive sent with normal generate requests (keeps model hot in
# VRAM between calls to reduce latency).  On shutdown we override this to 0
# so the model is evicted immediately.
# ---------------------------------------------------------------------------
OLLAMA_KEEP_ALIVE: str = "10m"


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
        self.window: Optional[tk.Tk] = None
        self.text_widget: Optional[scrolledtext.ScrolledText] = None
        self.prompt_text: Optional[scrolledtext.ScrolledText] = None
        self.response_text: Optional[scrolledtext.ScrolledText] = None
        self.status_label: Optional[ttk.Label] = None
        self.stats_labels: Dict[str, ttk.Label] = {}

        # Use plain Python bool instead of tk.BooleanVar to avoid
        # threading issues during garbage collection
        self._auto_scroll: bool = True

        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._shutdown_complete = threading.Event()

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

    def _toggle_auto_scroll(self):
        """Callback for the auto-scroll checkbutton."""
        self._auto_scroll = not self._auto_scroll

    def _run_gui(self):
        try:
            self.window = tk.Tk()
            self.window.title("LLM Correlator Debug Monitor")
            self.window.geometry("1200x800")
            self.window.configure(bg=self.COLORS['bg'])
            self._configure_dark_theme()

            # Handle the window-manager close button
            self.window.protocol("WM_DELETE_WINDOW", self._on_window_close)

            main_frame = ttk.Frame(self.window, style='Dark.TFrame')
            main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            stats_frame = ttk.LabelFrame(main_frame, text="Statistics", style='Dark.TLabelframe')
            stats_frame.pack(fill=tk.X, pady=(0, 5))

            self.stats_labels = {}
            stats = ["API Calls", "Total Tokens", "Avg Response Time", "Errors", "Context Size"]
            for i, stat in enumerate(stats):
                ttk.Label(stats_frame, text=f"{stat}:", style='Dark.TLabel').grid(
                    row=0, column=i * 2, padx=5, pady=5)
                self.stats_labels[stat] = ttk.Label(stats_frame, text="0", style='DarkValue.TLabel')
                self.stats_labels[stat].grid(row=0, column=i * 2 + 1, padx=5, pady=5)

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
                relief=tk.FLAT, borderwidth=0, padx=10, pady=10,
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
                relief=tk.FLAT, borderwidth=0, padx=10, pady=10,
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
                relief=tk.FLAT, borderwidth=0, padx=10, pady=10,
            )
            self.response_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

            button_frame = ttk.Frame(main_frame, style='Dark.TFrame')
            button_frame.pack(fill=tk.X, pady=(5, 0))

            ttk.Button(button_frame, text="Clear Log", command=self._clear_log,
                       style='Dark.TButton').pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="Save Log", command=self._save_log,
                       style='Dark.TButton').pack(side=tk.LEFT, padx=5)

            # Use a regular Checkbutton with command callback instead of variable
            # This avoids BooleanVar threading/GC issues
            self._auto_scroll_cb = ttk.Checkbutton(
                button_frame,
                text="Auto-scroll",
                command=self._toggle_auto_scroll,
                style='Dark.TCheckbutton',
            )
            self._auto_scroll_cb.pack(side=tk.LEFT, padx=5)
            # Set initial state to checked
            self._auto_scroll_cb.state(['selected'])

            self.status_label = ttk.Label(button_frame, text="● Ready", style='DarkValue.TLabel')
            self.status_label.pack(side=tk.RIGHT, padx=10)

            self.running = True
            self._process_queue()
            self.window.mainloop()
        except Exception as exc:
            self.running = False
            print(f"[ERROR] Debug monitor failed to start: {exc}")
        finally:
            # Signal that the GUI thread has fully exited
            self._shutdown_complete.set()

    def _on_window_close(self):
        """Handle user clicking the X button on the debug window."""
        self._do_destroy()

    def _process_queue(self):
        if not self.running:
            return
        try:
            while True:
                msg = self.message_queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        if self.running and self.window is not None:
            try:
                self.window.after(100, self._process_queue)
            except tk.TclError:
                # Window already destroyed
                self.running = False

    def _handle_message(self, msg: Dict[str, Any]):
        msg_type = msg.get("type", "info")
        if msg_type == "log":
            self._append_log(msg.get("text", ""), msg.get("tag", "info"))
        elif msg_type == "prompt":
            if self.prompt_text is not None:
                try:
                    self.prompt_text.delete(1.0, tk.END)
                    self.prompt_text.insert(tk.END, msg.get("text", ""))
                except tk.TclError:
                    pass
            self._append_log(f"▶ Sent prompt ({len(msg.get('text', ''))} chars)", "prompt")
        elif msg_type == "response":
            if self.response_text is not None:
                try:
                    self.response_text.delete(1.0, tk.END)
                    self.response_text.insert(tk.END, msg.get("text", ""))
                except tk.TclError:
                    pass
            self._append_log(f"◀ Received response ({len(msg.get('text', ''))} chars)", "response")
        elif msg_type == "stats":
            for key, value in msg.get("data", {}).items():
                if key in self.stats_labels:
                    try:
                        self.stats_labels[key].config(text=str(value))
                    except tk.TclError:
                        pass
        elif msg_type == "status":
            if self.status_label is not None:
                try:
                    self.status_label.config(text=msg.get("text", ""))
                except tk.TclError:
                    pass

    def _append_log(self, text: str, tag: str = "info"):
        if self.text_widget is not None:
            try:
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                self.text_widget.insert(tk.END, f"[{timestamp}] ", "timestamp")
                self.text_widget.insert(tk.END, f"{text}\n", tag)
                if self._auto_scroll:
                    self.text_widget.see(tk.END)
            except tk.TclError:
                # Widget destroyed
                pass

    def _clear_log(self):
        if self.text_widget is not None:
            try:
                self.text_widget.delete(1.0, tk.END)
                self._append_log("Log cleared", "info")
            except tk.TclError:
                pass

    def _save_log(self):
        if self.text_widget is not None:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"correlator_log_{timestamp}.txt"
                content = self.text_widget.get(1.0, tk.END)
                with open(filename, "w") as f:
                    f.write(content)
                self._append_log(f"Log saved to {filename}", "success")
            except tk.TclError:
                pass
            except IOError as e:
                self._append_log(f"Failed to save log: {e}", "error")

    def log(self, text: str, tag: str = "info"):
        if self.running:
            self.message_queue.put({"type": "log", "text": text, "tag": tag})

    def log_prompt(self, text: str):
        if self.running:
            self.message_queue.put({"type": "prompt", "text": text})

    def log_response(self, text: str):
        if self.running:
            self.message_queue.put({"type": "response", "text": text})

    def update_stats(self, stats: Dict[str, Any]):
        if self.running:
            self.message_queue.put({"type": "stats", "data": stats})

    def set_status(self, text: str):
        if self.running:
            self.message_queue.put({"type": "status", "text": text})

    def shutdown(self):
        """Close the Tk window and stop the GUI loop."""
        if not self.running:
            return

        self.running = False

        try:
            if self.window is not None:
                # Schedule destroy on the Tk thread (Tk is not thread-safe)
                self.window.after(0, self._do_destroy)
        except Exception:
            pass

        # Wait for the GUI thread to finish with timeout
        self._shutdown_complete.wait(timeout=3.0)

        # Ensure thread is joined
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

    def _do_destroy(self):
        """Called on the Tk mainloop thread to actually destroy the window.

        Must clean up all Tk resources while still on the Tk thread to avoid
        threading errors during garbage collection.
        """
        self.running = False

        try:
            # Clear all widget references BEFORE destroying window
            # This allows them to be GC'd while Tk is still valid
            self.text_widget = None
            self.prompt_text = None
            self.response_text = None
            self.status_label = None
            self.stats_labels.clear()

            # Force a GC pass while Tk is still alive to clean up any
            # Tk-associated objects (prevents "main thread not in main loop")
            gc.collect()

            if self.window is not None:
                try:
                    self.window.quit()
                except Exception:
                    pass
                try:
                    self.window.destroy()
                except Exception:
                    pass
                self.window = None

        except Exception:
            pass
        finally:
            self._shutdown_complete.set()


class RollingContextManager:
    """Manages rolling context window for the LLM with strict budget enforcement."""

    def __init__(
        self,
        max_context_tokens: int = OLLAMA_CONTEXT_WINDOW_TOKENS,
        max_response_tokens: int = OLLAMA_MAX_RESPONSE_TOKENS,
    ):
        self.max_context_tokens = max_context_tokens
        self.max_response_tokens = max_response_tokens
        self.max_prompt_tokens = max_context_tokens - max_response_tokens

        self.chars_per_token = OLLAMA_CHARS_PER_TOKEN
        self.tokens_per_correlation = OLLAMA_TOKENS_PER_CORRELATION
        self.token_estimate_buffer = OLLAMA_TOKEN_ESTIMATE_BUFFER
        self.response_json_overhead = OLLAMA_RESPONSE_JSON_OVERHEAD
        self.max_transmission_batch = OLLAMA_MAX_TRANSMISSION_BATCH
        self.adsb_prompt_ratio = OLLAMA_ADSB_PROMPT_RATIO

        self.system_prompt: str = ""
        self.system_prompt_tokens: int = 0

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt
        self.system_prompt_tokens = self._estimate_tokens(prompt)

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text) / self.chars_per_token) + self.token_estimate_buffer

    def calculate_max_transmissions(self, num_adsb: int) -> int:
        """Calculate how many transmissions we can safely process."""
        available_response = self.max_response_tokens - self.response_json_overhead
        max_tx = available_response // self.tokens_per_correlation
        return min(max_tx, self.max_transmission_batch)

    def build_context_prompt(
        self,
        adsb_data: Sequence[ADSBContact],
        transmissions: Sequence[ATCTransmission],
        format_adsb_func,
        format_transmissions_func,
    ) -> tuple[str, int, int]:
        """Build prompt that fits within context window."""
        available = self.max_prompt_tokens - self.system_prompt_tokens

        template = self._get_analysis_template()
        template_tokens = self._estimate_tokens(template)
        available -= template_tokens

        adsb_budget = int(available * self.adsb_prompt_ratio)
        tx_budget = available - adsb_budget

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

        max_tx = self.calculate_max_transmissions(len(adsb_included))

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

        prompt = (
            f"{self.system_prompt}\n\n"
            f"{template.format(adsb_data=adsb_text, tx_data=tx_text, num_tx=len(tx_included))}"
        )

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

    def __init__(
        self,
        max_adsb_contacts: int = LLM_MAX_ADSB_CONTACTS,
        max_transmissions: int = LLM_MAX_TRANSMISSIONS,
        preview_chars: int = OLLAMA_TRANSMISSION_PREVIEW_CHARS,
    ):
        self.max_adsb = max_adsb_contacts
        self.max_tx = max_transmissions
        self.preview_chars = preview_chars

    def build_system_prompt(self) -> str:
        return (
            "ATC correlation analyst. Match transmissions to ADS-B using fuzzy logic.\n"
            "\n"
            "AIRLINE CODES: DAL=Delta UAL=United AAL=American SWA=Southwest "
            "JBU=JetBlue ASA=Alaska SKW=SkyWest ENY=Envoy RPA=Republic FFT=Frontier NKS=Spirit\n"
            "\n"
            "FUZZY MATCH RULES:\n"
            '- Numbers: DAL2617="Delta 26 17"="Delta 2617"="Delta 2 6 1 7"\n'
            "- Phonetic: tree=3 fife=5 niner=9\n"
            "- Errors: 5↔9 B↔D↔P M↔N common, off-by-one OK\n"
            '- Partial: "Delta 617"→DAL2617/DAL1617, suffix "...17"→*17\n'
            '- Transcription: "delta"→"data"/"dealt", split words OK\n'
            "\n"
            "CONFIDENCE: exact=0.95, 1-digit-off=0.80, partial=0.70, fuzzy=0.60, unclear=0.30\n"
            "\n"
            "MATCH if conf≥0.50. ALERT only if: extraction≥0.80, definitely absent, alert_conf≥0.70\n"
            "\n"
            "MILITARY: REACH RCH VIPER EAGLE HAMMER KING RESCUE EVAC DUKE\n"
            "\n"
            "OUTPUT (keep reasoning SHORT):\n"
            '{"correlations":[{"transmission_id":0,"extracted_identifier":"text",'
            '"extraction_confidence":0.8,"matched_icao":"ICAO/NO_MATCH/UNCLEAR",'
            '"matched_callsign":"CS","match_confidence":0.7,"reasoning":"brief","flags":[]}],'
            '"alerts":[],"summary":"brief"}'
        )

    def _format_adsb(self, contacts: Sequence[ADSBContact]) -> str:
        lines: List[str] = []
        for contact in contacts:
            cs = contact.callsign or "----"
            lines.append(
                f"{contact.icao} {cs:8} {contact.altitude}ft "
                f"{contact.heading:03}° {contact.speed}kt"
            )
        return "\n".join(lines) if lines else "(none)"

    def _format_transmissions(self, txs: Sequence[ATCTransmission]) -> str:
        lines: List[str] = []
        for idx, tx in enumerate(txs):
            if len(tx.text) > self.preview_chars:
                text = tx.text[: self.preview_chars] + "..."
            else:
                text = tx.text
            lines.append(f'[{idx}] {tx.channel_name}: "{text}"')
        return "\n".join(lines) if lines else "(none)"


class OllamaCorrelator:
    """Query an Ollama-hosted model to correlate ATC transmissions.

    Supports the context-manager protocol so the model is always unloaded
    cleanly from VRAM::

        with OllamaCorrelator() as correlator:
            result = correlator.correlate(adsb_data, transmissions)
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        max_adsb_contacts: int = LLM_MAX_ADSB_CONTACTS,
        max_transmissions: int = LLM_MAX_TRANSMISSIONS,
        request_timeout: int = OLLAMA_REQUEST_TIMEOUT,
        enable_debug_monitor: bool = OLLAMA_ENABLE_DEBUG_MONITOR,
        keep_alive: str = OLLAMA_KEEP_ALIVE,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive
        self._closed = False
        self.context_builder = ContextBuilder(
            max_adsb_contacts=max_adsb_contacts,
            max_transmissions=max_transmissions,
        )
        self.rolling_context = RollingContextManager(
            max_context_tokens=OLLAMA_CONTEXT_WINDOW_TOKENS,
            max_response_tokens=OLLAMA_MAX_RESPONSE_TOKENS,
        )
        self.request_timeout = request_timeout

        self.stats: Dict[str, Any] = {
            "API Calls": 0,
            "Total Tokens": 0,
            "Avg Response Time": "0.0s",
            "Errors": 0,
            "Context Size": f"0/{self.rolling_context.max_context_tokens}",
        }
        self._response_times: List[float] = []

        # ---- debug monitor setup ----
        self.debug_monitor: Optional[DebugMonitor] = None
        if enable_debug_monitor:
            if not HAS_TK:
                if sys.platform.startswith("linux"):
                    print(
                        "[WARN] Tkinter not available; install python3-tk "
                        "to enable the debug monitor."
                    )
                else:
                    print("[WARN] Tkinter not available; debug monitor disabled.")
            elif not _has_display():
                print(
                    "[WARN] No GUI display detected (DISPLAY/WAYLAND_DISPLAY); "
                    "debug monitor disabled."
                )
            else:
                self.debug_monitor = DebugMonitor()
                time.sleep(OLLAMA_DEBUG_MONITOR_DELAY)

        # ---- system prompt ----
        system_prompt = self.context_builder.build_system_prompt()
        self.rolling_context.set_system_prompt(system_prompt)

        # ---- safety-net: always unload on interpreter exit ----
        atexit.register(self.close)

        self._log("Correlator initialized", "info")
        self._log(f"Model: {model}", "info")
        self._log(f"Keep-alive: {keep_alive}", "info")
        self._log(
            f"System prompt: ~{self.rolling_context.system_prompt_tokens} tokens", "info"
        )
        self._log(f"Max prompt: {self.rolling_context.max_prompt_tokens} tokens", "info")
        self._log(
            f"Max response: {self.rolling_context.max_response_tokens} tokens", "info"
        )

    # ------------------------------------------------------------------
    # Context-manager & lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> "OllamaCorrelator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def __del__(self) -> None:
        # Last-resort fallback; do not rely on this alone.
        try:
            self.close()
        except Exception:
            pass

    def unload_model(self) -> None:
        """Ask Ollama to evict the model from VRAM immediately.

        Sending ``keep_alive: 0`` tells Ollama to unload the model as soon
        as the (empty) request completes.
        """
        try:
            self._log(f"Unloading model '{self.model}' from VRAM...", "warning")
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "",
                    "keep_alive": 0,
                },
                timeout=15,
            )
            if resp.ok:
                self._log(f"Model '{self.model}' unloaded from VRAM", "success")
            else:
                self._log(
                    f"Model unload returned {resp.status_code}: {resp.text[:200]}",
                    "warning",
                )
        except requests.exceptions.ConnectionError:
            self._log(
                "Ollama not reachable during unload (may already be stopped)",
                "warning",
            )
        except Exception as exc:
            self._log(f"Error unloading model: {exc}", "error")

    def close(self) -> None:
        """Gracefully shut down: unload model from VRAM, close debug monitor."""
        if self._closed:
            return
        self._closed = True

        self._log("Shutting down correlator...", "warning")

        # Unload the model first (while we can still log)
        self.unload_model()

        # Tear down the debug monitor
        if self.debug_monitor is not None:
            self.debug_monitor.shutdown()
            self.debug_monitor = None

        # De-register atexit so it doesn't fire twice
        try:
            atexit.unregister(self.close)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Logging / stats helpers
    # ------------------------------------------------------------------

    def _log(self, message: str, tag: str = "info") -> None:
        if self.debug_monitor is not None:
            self.debug_monitor.log(message, tag)
        else:
            print(f"[{tag.upper()}] {message}")

    def _update_stats(self) -> None:
        if self.debug_monitor is not None:
            self.debug_monitor.update_stats(self.stats)

    def _set_status(self, status: str) -> None:
        if self.debug_monitor is not None:
            self.debug_monitor.set_status(status)

    # ------------------------------------------------------------------
    # Core correlation
    # ------------------------------------------------------------------

    def correlate(
        self,
        adsb_data: Sequence[ADSBContact],
        transmissions: Sequence[ATCTransmission],
    ) -> dict:
        if self._closed:
            return {
                "error": "Correlator has been shut down",
                "correlations": [],
                "alerts": [],
                "summary": "Shut down",
            }

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
        self.stats["Context Size"] = (
            f"{prompt_tokens}+{self.rolling_context.max_response_tokens}"
            f"/{self.rolling_context.max_context_tokens}"
        )

        if total_budget > self.rolling_context.max_context_tokens:
            self._log(
                f"WARNING: Budget overflow! {total_budget} > "
                f"{self.rolling_context.max_context_tokens}",
                "error",
            )

        self._log(
            f"Prompt: {num_adsb} contacts, {num_tx} transmissions, ~{prompt_tokens} tokens",
            "info",
        )
        self._log(
            f"Expected response: ~{num_tx * self.rolling_context.tokens_per_correlation} tokens",
            "info",
        )

        if self.debug_monitor is not None:
            self.debug_monitor.log_prompt(prompt)

        try:
            self._log(f"Sending request to {self.model}...", "info")

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": self.keep_alive,
                    "options": {
                        "temperature": OLLAMA_TEMPERATURE,
                        "num_predict": self.rolling_context.max_response_tokens,
                        "top_p": OLLAMA_TOP_P,
                        "num_ctx": self.rolling_context.max_context_tokens,
                        "repeat_penalty": OLLAMA_REPEAT_PENALTY,
                    },
                },
                timeout=self.request_timeout,
            )
            response.raise_for_status()

            elapsed = time.time() - start_time
            self._response_times.append(elapsed)
            if len(self._response_times) > OLLAMA_RESPONSE_TIME_WINDOW:
                self._response_times.pop(0)
            avg_time = sum(self._response_times) / len(self._response_times)
            self.stats["Avg Response Time"] = f"{avg_time:.1f}s"

            result = response.json()
            response_text = result.get("response", "")

            if self.debug_monitor is not None:
                self.debug_monitor.log_response(response_text)

            eval_count = result.get("eval_count", 0)
            prompt_eval_count = result.get("prompt_eval_count", 0)
            self.stats["Total Tokens"] += eval_count + prompt_eval_count

            self._log(
                f"Response: {elapsed:.1f}s "
                f"(prompt: {prompt_eval_count}, response: {eval_count} tokens)",
                "response",
            )

            # Check for truncation
            if eval_count >= (
                self.rolling_context.max_response_tokens - OLLAMA_RESPONSE_SAFETY_MARGIN
            ):
                self._log(
                    "WARNING: Response likely truncated (hit token limit)", "warning"
                )
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
                    alert
                    for alert in parsed["alerts"]
                    if alert.get("confidence", 0) >= OLLAMA_ALERT_CONFIDENCE_THRESHOLD
                ]
                filtered = original_count - len(parsed["alerts"])
                if filtered > 0:
                    self._log(f"Filtered {filtered} low-confidence alerts", "info")

            if "correlations" in parsed:
                matches = sum(
                    1
                    for c in parsed["correlations"]
                    if c.get("matched_icao") not in ["NO_MATCH", "UNCLEAR", None]
                )
                self._log(
                    f"Matches: {matches}/{len(parsed['correlations'])}", "success"
                )

            if parsed.get("alerts"):
                for alert in parsed["alerts"]:
                    self._log(
                        f"ALERT: {alert.get('type')} - {alert.get('callsign')} "
                        f"({alert.get('confidence', 0):.2f})",
                        "warning",
                    )

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

    # ------------------------------------------------------------------
    # JSON repair / parse
    # ------------------------------------------------------------------

    def _attempt_json_repair(self, text: str) -> str:
        text = text.rstrip()

        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")

        if open_braces > 0 or open_brackets > 0:
            self._log(
                f"Repairing JSON: missing {open_braces} braces, "
                f"{open_brackets} brackets",
                "warning",
            )

            repair_point = len(text)
            for pattern in ['"}', "']", "},", "],", "}"]:
                pos = text.rfind(pattern)
                if pos > 0:
                    repair_point = min(repair_point, pos + len(pattern))

            if repair_point < len(text):
                text = text[:repair_point]

            # Re-count after trim
            open_braces = text.count("{") - text.count("}")
            open_brackets = text.count("[") - text.count("]")
            text += "]" * max(open_brackets, 0) + "}" * max(open_braces, 0)

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
            if hasattr(e, "pos") and e.pos:
                start_ctx = max(0, e.pos - 30)
                end_ctx = min(len(text), e.pos + 30)
                self._log(f"Error context: ...{text[start_ctx:end_ctx]}...", "error")

        return {"error": "Failed to parse LLM response", "raw": text}


# -----------------------------------------------------------------------
# Module-level helpers
# -----------------------------------------------------------------------


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


def _has_display() -> bool:
    if sys.platform.startswith("linux"):
        return bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
    return True


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