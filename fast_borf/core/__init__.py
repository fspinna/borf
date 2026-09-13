"""The computing steps behind BORF, usable on their own.

For one signal (without missing values) and its timestamps:

- segment_means: z-normalized segment means of every window (the PAA step)
- breakpoints and discretize: segment means to SAX symbols
- sax: both steps; sax_words: the same words directly as integers
- window_positions: the points covered by each segment of each window
- encode_words and decode_words: SAX words to integers and back

For a panel of series, panel_words gives the word at every window of every
signal and transform_sax_patterns counts them.
"""

from fast_borf.core.sax import (
    breakpoints,
    discretize,
    sax,
    sax_words,
    segment_means,
    window_positions,
)
from fast_borf.core.transform import panel_words, transform_sax_patterns
from fast_borf.core.words import decode_words, encode_words

__all__ = [
    "breakpoints",
    "decode_words",
    "discretize",
    "encode_words",
    "panel_words",
    "sax",
    "sax_words",
    "segment_means",
    "transform_sax_patterns",
    "window_positions",
]
