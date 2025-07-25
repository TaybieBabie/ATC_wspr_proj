import threading
import sys
from datetime import datetime
from enum import Enum
import time

from tqdm import tqdm


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    SUCCESS = 2
    WARNING = 3
    ERROR = 4


class LoggingTqdm(tqdm):
    """tqdm wrapper that integrates with our logger"""

    def __init__(self, *args, **kwargs):
        # Save current progress state
        with logger.lock:
            was_active = logger.progress_active
            if was_active:
                logger._clear_progress()

        super().__init__(*args, **kwargs)

        # Restore previous progress if needed
        if was_active:
            logger._restore_progress()

    def display(self, msg=None, pos=None):
        """Override display to use our logger for progress updates"""
        if not msg:
            msg = self.__str__()

        with logger.lock:
            was_active = logger.progress_active
            if was_active:
                logger._clear_progress()

            # Display the tqdm progress
            logger.progress(msg)


print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    """Thread-safe print function that respects progress bars"""
    with logger.lock:
        was_active = logger.progress_active
        if was_active:
            logger._clear_progress()

        print(*args, **kwargs)

        if was_active:
            logger._restore_progress()

class ConsoleLogger:
    """Thread-safe console logger with progress bar handling"""

    def __init__(self, min_level=LogLevel.INFO):
        self.lock = threading.Lock()
        self.min_level = min_level
        self.progress_active = False
        self.last_progress_line = ""

    def _clear_progress(self):
        """Clear the current progress line if one exists"""
        if self.progress_active:
            # Move cursor to beginning of line and clear
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            self.progress_active = False

    def _restore_progress(self):
        """Restore the progress bar after printing a message"""
        if self.last_progress_line:
            sys.stdout.write(self.last_progress_line)
            sys.stdout.flush()
            self.progress_active = True

    def log(self, message, level=LogLevel.INFO, emoji=None):
        """Log a message with the specified level"""
        if level.value < self.min_level.value:
            return

        with self.lock:
            self._clear_progress()

            timestamp = datetime.now().strftime("%H:%M:%S")
            prefix = f"[{timestamp}]"

            if emoji:
                prefix += f" {emoji}"

            # Color based on level
            if level == LogLevel.ERROR:
                color_code = "\033[91m"  # Red
            elif level == LogLevel.WARNING:
                color_code = "\033[93m"  # Yellow
            elif level == LogLevel.SUCCESS:
                color_code = "\033[92m"  # Green
            elif level == LogLevel.DEBUG:
                color_code = "\033[94m"  # Blue
            else:
                color_code = "\033[0m"  # Default

            reset_code = "\033[0m"

            print(f"{color_code}{prefix} {message}{reset_code}")

            self._restore_progress()

    def debug(self, message, emoji="🔍"):
        self.log(message, LogLevel.DEBUG, emoji)

    def info(self, message, emoji="ℹ️"):
        self.log(message, LogLevel.INFO, emoji)

    def success(self, message, emoji="✅"):
        self.log(message, LogLevel.SUCCESS, emoji)

    def warning(self, message, emoji="⚠️"):
        self.log(message, LogLevel.WARNING, emoji)

    def error(self, message, emoji="❌"):
        self.log(message, LogLevel.ERROR, emoji)

    def progress(self, line):
        """Update progress bar"""
        with self.lock:
            self._clear_progress()
            sys.stdout.write(line)
            sys.stdout.flush()
            self.progress_active = True
            self.last_progress_line = line

    def section(self, title, emoji="📋"):
        """Print a section header"""
        with self.lock:
            self._clear_progress()
            print("\n" + "=" * 60)
            print(f"{emoji} {title}")
            print("=" * 60)
            self._restore_progress()


# Global logger instance
logger = ConsoleLogger()


# Export convenience functions
def debug(message, emoji="🔍"):
    logger.debug(message, emoji)


def info(message, emoji="ℹ️"):
    logger.info(message, emoji)


def success(message, emoji="✅"):
    logger.success(message, emoji)


def warning(message, emoji="⚠️"):
    logger.warning(message, emoji)


def error(message, emoji="❌"):
    logger.error(message, emoji)


def progress(line):
    logger.progress(line)


def section(title, emoji="📋"):
    logger.section(title, emoji)


class ProgressBar:
    """Custom progress bar that works with the logger"""

    def __init__(self, total, desc="Processing", unit="it"):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
        self._update(0)

    def _format_time(self, seconds):
        """Format time in seconds to MM:SS"""
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes:02d}:{seconds:02d}"

    def _update(self, n):
        """Update progress bar display"""
        self.current += n
        percentage = min(100, int(100 * self.current / self.total))

        # Only update display 4 times per second to avoid flicker
        current_time = time.time()
        if current_time - self.last_update < 0.25 and percentage < 100:
            return

        self.last_update = current_time

        # Calculate elapsed and remaining time
        elapsed = current_time - self.start_time
        if percentage > 0:
            remaining = elapsed * (100 - percentage) / percentage
        else:
            remaining = 0

        # Create progress bar
        bar_length = 30
        filled_length = int(bar_length * percentage / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        # Format line
        line = f"\r{self.desc}: [{bar}] {percentage}% | {self._format_time(elapsed)}<{self._format_time(remaining)}"

        # Update display
        progress(line)

        # Print newline on completion
        if percentage >= 100:
            print()

    def update(self, n=1):
        """Update progress by n units"""
        self._update(n)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.current < self.total:
            self._update(self.total - self.current)
