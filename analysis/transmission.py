from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from tracking.adsb_tracker import Aircraft


@dataclass
class Transmission:
    """Represents a single ATC transmission with extracted metadata."""

    timestamp: datetime
    text: str
    callsigns: List[str] = field(default_factory=list)
    altitudes: List[int] = field(default_factory=list)
    headings: List[int] = field(default_factory=list)
    frequencies: List[str] = field(default_factory=list)
    aircraft: Optional[Aircraft] = None
    possible_aircraft: List[Aircraft] = field(default_factory=list)
    confidence: float = 0.0
