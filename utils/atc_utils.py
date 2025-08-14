import re

# Shared regex patterns for ATC-related parsing
# Matches FAA "N" numbers and common airline callsign formats like AAA123
CALLSIGN_PATTERN = r'\b(?:[A-Z]{3}\d{1,4}|N\d{1,5}[A-Z]{0,2})\b'
CALLSIGN_REGEX = re.compile(CALLSIGN_PATTERN)
