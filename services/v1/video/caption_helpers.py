# caption_helpers.py
# GPL-2.0-or-later — 2025 Stephen G. Pope

from __future__ import annotations
import re
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ass_timestamp(seconds: float) -> str:
    """Convert seconds → ASS h:mm:ss.cc (centiseconds)."""
    cs = int(round(seconds * 100))               # centiseconds
    h, cs = divmod(cs, 360_000)                  # 360000 cs  =  1 h
    m, cs = divmod(cs,  6_000)                   # 6000 cs   =  1 min
    s, cs = divmod(cs,    100)                   # 100 cs    =  1 s
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"     # 0:00:00.00


def _make_style_line(name: str, opts: Dict) -> str:
    """Return a single 'Style: …' line (ASS v4+)."""
    fmt_defaults = dict(
        Fontname="Roboto",
        Fontsize=48,
        PrimaryColour="&H00FFFFFF",   # white
        SecondaryColour="&H000000FF", # blue (karaoke)
        OutlineColour="&H00000000",   # black
        BackColour="&H64000000",      # 60% opaque black
        Bold=0, Italic=0, Underline=0, StrikeOut=0,
        ScaleX=100, ScaleY=100, Spacing=0, Angle=0,
        BorderStyle=1, Outline=2, Shadow=0,
        Alignment=2,                 # 2 = bottom-centre
        MarginL=60, MarginR=60, MarginV=40,
        Encoding=1,                  # UTF-8
    )
    fmt_defaults.update(opts)
    values = ",".join(str(fmt_defaults[k]) for k in [
        "Fontname", "Fontsize",
        "PrimaryColour", "SecondaryColour", "OutlineColour", "BackColour",
        "Bold", "Italic", "Underline", "StrikeOut",
        "ScaleX", "ScaleY", "Spacing", "Angle",
        "BorderStyle", "Outline", "Shadow",
        "Alignment", "MarginL", "MarginR", "MarginV",
        "Encoding"
    ])
    return f"Style: {name},{values}"


def _apply_replacements(text: str, replace_map: Dict[str, str]) -> str:
    """Do straightforward textual replacements (case-sensitive)."""
    for find, repl in replace_map.items():
        text = text.replace(find, repl)
    return text


# ---------------------------------------------------------------------------
# Main public helper
# ---------------------------------------------------------------------------

def process_subtitle_events(
    transcription: Dict,
    style: str,
    settings: Dict,
    replace_map: Dict[str, str],
    resolution: Tuple[int, int],
) -> str:
    """
    Build an ASS file from Whisper-like transcription dict.

    Parameters
    ----------
    transcription : dict
        As returned by `generate_transcription()` or the SRT shim in the pipeline.
        Must contain a key ``"segments"`` with ``start`` / ``end`` / ``text``.
    style : str
        One of ``"classic"`` (default white on black) or ``"modern"``
        (semi-transparent box with rounded corners).  Extend at will!
    settings : dict
        Optional overrides, e.g.
          {
            "font": "Inter",
            "font_size": 42,
            "primary_colour": "&H00FFFF00",   # yellow
            "alignment": 2,                   # bottom-centre
          }
    replace_map : dict
        ``{"bad": "good", …}`` replacements applied to each line.
    resolution : (w, h)
        Currently unused (kept for future per-resolution tweaks).
    """
    # ---------------------------------------------------------------------
    # 1. Header — Script Info
    # ---------------------------------------------------------------------
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {resolution[0]}",
        f"PlayResY: {resolution[1]}",
        "YCbCr Matrix: TV.709",
        "",
    ]

    # ---------------------------------------------------------------------
    # 2. Styles
    # ---------------------------------------------------------------------
    font        = settings.get("font",        "Roboto")
    font_size   = settings.get("font_size",   48)
    pri_colour  = settings.get("primary_colour", "&H00FFFFFF")   # white
    outline     = settings.get("outline",     2)
    shadow      = settings.get("shadow",      0)
    alignment   = settings.get("alignment",   2)                 # bottom-centre
    margin_v    = settings.get("margin_v",    40)

    if style == "modern":
        back_colour = settings.get("back_colour", "&H32000000")   # 20% black
    else:  # classic
        back_colour = settings.get("back_colour", "&H64000000")   # 60% black

    style_opts = dict(
        Fontname=font,
        Fontsize=font_size,
        PrimaryColour=pri_colour,
        BackColour=back_colour,
        Outline=outline,
        Shadow=shadow,
        Alignment=alignment,
        MarginV=margin_v,
    )

    styles_section = [
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
        "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        _make_style_line("Default", style_opts),
        "",
    ]

    # ---------------------------------------------------------------------
    # 3. Events  ----------------------------------------------------------
    # ---------------------------------------------------------------------
    events_header = [
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    dialogue_lines: List[str] = []

    for seg in transcription.get("segments", []):
        start  = _ass_timestamp(seg["start"])
        end    = _ass_timestamp(seg["end"])
        text   = seg["text"].strip().replace("\n", " ")
        text   = re.sub(r"\s+", " ", text)          # collapse whitespace
        text   = _apply_replacements(text, replace_map)

        # -- Word-level karaoke effect (optional) --------------------------
        # If we *do* have word timings and want karaoke, you can extend here:
        # if seg.get("words") and style == "karaoke":
        #     parts = []
        #     for w in seg["words"]:
        #         dur_cs = int(round((w["end"] - w["start"]) * 100))
        #         parts.append(rf"\k{dur_cs}{w['word']}")
        #     text = "{" + "".join(parts) + "}"

        dialogue_lines.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
        )

    # ---------------------------------------------------------------------
    # 4. Glue everything and return
    # ---------------------------------------------------------------------
    ass_text = "\n".join(header + styles_section + events_header + dialogue_lines)
    return ass_text
