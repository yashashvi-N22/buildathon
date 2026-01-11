import io
import tempfile
import os
from typing import Dict, Optional, Tuple

import numpy as np
import re
import streamlit as st

# Optional plotting: prefer plotly, fallback to matplotlib
try:
    import plotly.graph_objects as go  # type: ignore
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False
    #import matplotlib.pyplot as plt

# Audio processing
import librosa
import soundfile as sf

# Speech to text
import speech_recognition as sr

# Grammar check
import language_tool_python


# ----------------------------
# Utility: Scoring Heuristics
# ----------------------------
def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return float(max(low, min(high, value)))


def compute_voice_metrics(y: np.ndarray, sr_hz: int) -> Dict[str, float]:
    """
    Compute simple voice metrics from waveform using librosa.

    Returns keys:
    - pitch_variation: standard deviation of fundamental frequency (Hz)
    - rms_stability: inverse of RMS variance (higher is steadier)
    - speaking_rate_wpm: rough words-per-minute estimate from syllable-like peaks
    - duration_sec: audio duration in seconds
    - clarity_ratio: voiced_time / total_time (proxy for articulation/steadiness)
    """
    if y.size == 0:
        return {
            "pitch_variation": 0.0,
            "rms_stability": 0.0,
            "speaking_rate_wpm": 0.0,
            "duration_sec": 0.0,
            "clarity_ratio": 0.0,
        }

    # Ensure mono
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    duration_sec = max(1e-6, y.shape[0] / sr_hz)

    # RMS for amplitude stability (lower variance -> more stable)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_var = float(np.var(rms))
    # Convert to stability in [0,1] via inverse mapping with soft norm
    rms_stability = 1.0 / (1.0 + 50.0 * rms_var)

    # Pitch track using pyin when possible, fallback to piptrack
    try:
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
    except Exception:
        S = np.abs(librosa.stft(y))
        pitches, mags = librosa.piptrack(S=S, sr=sr_hz)
        pitches = pitches[mags.argmax(axis=0), np.arange(mags.shape[1])] if mags.size else np.array([])
        f0 = pitches

    if f0 is None:
        f0 = np.array([])

    valid_f0 = f0[np.isfinite(f0)]
    valid_f0 = valid_f0[valid_f0 > 0]
    pitch_variation = float(np.std(valid_f0)) if valid_f0.size else 0.0

    # Voiced frames ratio as clarity proxy
    if f0 is not None and np.asarray(f0).size:
        voiced_mask = np.isfinite(f0) & (np.asarray(f0) > 0)
        clarity_ratio = float(np.mean(voiced_mask))
    else:
        # Energy-based proxy
        energy = rms
        clarity_ratio = float(np.mean(energy > (energy.mean() * 0.5))) if energy.size else 0.0

    # Speaking rate rough estimate:
    # Count energy peaks as syllable-like events and divide by average syllables per word (~1.4)
    try:
        energy_smooth = librosa.effects.hpss(y)[0]
        env = librosa.onset.onset_strength(y=energy_smooth, sr=sr_hz)
        peaks = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.2, wait=5)
        n_syllables = max(1, len(peaks))
        estimated_words = n_syllables / 1.4
        speaking_rate_wpm = float((estimated_words / duration_sec) * 60.0)
    except Exception:
        # If onset/peak detection fails (e.g., pure tones), set to 0
        speaking_rate_wpm = 0.0

    return {
        "pitch_variation": pitch_variation,
        "rms_stability": float(rms_stability),
        "speaking_rate_wpm": speaking_rate_wpm,
        "duration_sec": duration_sec,
        "clarity_ratio": clarity_ratio,
    }


def voice_confidence_score(metrics: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """
    Heuristic confidence score considering steadiness (RMS), clarity (voiced ratio),
    and pitch variation (moderate variation preferred). Returns (score_percent, submetrics_scaled).
    """
    rms_stability = metrics.get("rms_stability", 0.0)  # 0..1 (higher is steadier)
    clarity = metrics.get("clarity_ratio", 0.0)  # 0..1 (more voiced frames)
    pitch_var = metrics.get("pitch_variation", 0.0)
    speaking_rate_wpm = metrics.get("speaking_rate_wpm", 0.0)

    # Pitch variation: penalize extremes; optimal band ~ 20-120 Hz std
    if pitch_var <= 0:
        pitch_component = 0.5
    else:
        if pitch_var < 20:
            pitch_component = 0.4 + 0.6 * (pitch_var / 20.0)
        elif pitch_var > 120:
            # linearly decay after 120
            pitch_component = max(0.2, 1.0 - (pitch_var - 120) / 200.0)
        else:
            pitch_component = 1.0

    # Speaking rate optimal around 110-160 wpm
    if speaking_rate_wpm <= 0:
        rate_component = 0.5
    else:
        ideal_low, ideal_high = 110.0, 160.0
        if speaking_rate_wpm < ideal_low:
            rate_component = max(0.3, speaking_rate_wpm / ideal_low)
        elif speaking_rate_wpm > ideal_high:
            rate_component = max(0.3, ideal_high / speaking_rate_wpm)
        else:
            rate_component = 1.0

    # Weighted combination
    weighted = (
        0.4 * rms_stability +
        0.3 * clarity +
        0.2 * pitch_component +
        0.1 * rate_component
    )
    score = clamp(100.0 * weighted)

    return score, {
        "steadiness_pct": clamp(100.0 * rms_stability),
        "clarity_pct": clamp(100.0 * clarity),
        "pitch_quality_pct": clamp(100.0 * pitch_component),
        "rate_match_pct": clamp(100.0 * rate_component),
    }


def transcribe_with_speech_recognition(wav_bytes: bytes) -> Tuple[str, Optional[float]]:
    """Transcribe audio using Google Web Speech via SpeechRecognition.
    Returns (text, confidence) where confidence may be None (API often omits).
    """
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(wav_bytes)) as source:
        audio_data = recognizer.record(source)
    try:
        # Primary: Google Web Speech
        result = recognizer.recognize_google(audio_data, show_all=True)
        if isinstance(result, dict) and result.get("results"):
            alts = result["results"][0].get("alternatives", [])
            if alts:
                text = alts[0].get("transcript", "").strip()
                conf = alts[0].get("confidence")
                return text, conf
        # Fallback simple
        text_simple = recognizer.recognize_google(audio_data)
        return text_simple, None
    except (sr.UnknownValueError, sr.RequestError):
        # Try offline CMU Sphinx if available
        try:
            text_sphinx = recognizer.recognize_sphinx(audio_data)
            return text_sphinx.strip(), None
        except Exception:
            return "", None


def is_wav_bytes(data: bytes) -> bool:
    """Heuristically detect if bytes look like a WAV file (RIFF/WAVE header)."""
    if not data or len(data) < 12:
        return False
    try:
        return data[0:4] == b"RIFF" and data[8:12] == b"WAVE"
    except Exception:
        return False


def load_audio_and_to_wav_bytes(file_bytes: bytes, target_sr: int = 16000, filename_suffix: str = ".tmp") -> bytes:
    """Load arbitrary audio bytes via librosa and re-encode as mono WAV bytes.
    filename_suffix helps decoders (e.g., ".mp3", ".m4a") when magic detection is unreliable.
    """
    # Fast path: already WAV
    if is_wav_bytes(file_bytes):
        return file_bytes

    tmp_path = None
    try:
        suffix = filename_suffix if filename_suffix and filename_suffix.startswith(".") else ".tmp"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
            tmp_in.write(file_bytes)
            tmp_path = tmp_in.name
        # Try librosa decode first (handles mp3/m4a via ffmpeg/audioread)
        try:
            y, sr_hz = librosa.load(tmp_path, sr=target_sr, mono=True)
        except Exception:
            # Fallback to soundfile for formats it supports (wav/flac/ogg)
            data, sr_hz = sf.read(tmp_path, always_2d=False)
            if isinstance(data, np.ndarray):
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                # Resample if needed
                if sr_hz != target_sr:
                    y = librosa.resample(y=data.astype(np.float32), orig_sr=sr_hz, target_sr=target_sr)
                else:
                    y = data.astype(np.float32)
            else:
                raise
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    with io.BytesIO() as buf:
        sf.write(buf, y, target_sr, format="WAV")
        return buf.getvalue()


def grammar_analysis(text: str) -> Tuple[float, int, int, list]:
    """Analyze grammar using language_tool_python with robust fallbacks.
    Returns (score_pct, num_errors, len_chars, matches).
    """
    text = (text or "").strip()
    if not text:
        return 0.0, 0, 0, []

    def _score_from_matches(_matches: list, _len_chars: int) -> float:
        # Heuristic scoring: penalize by error density and severity
        num_err = len(_matches)
        if _len_chars <= 0:
            return 0.0
        severity_penalty_local = 0.0
        for m in _matches:
            cat = getattr(m, "ruleIssueType", "") or getattr(getattr(m, "rule", None), "issueType", "")
            cat_l = str(cat).lower()
            if cat_l in {"grammar", "misspelling"}:
                severity_penalty_local += 1.5
            elif cat_l in {"style"}:
                severity_penalty_local += 0.75
            else:
                severity_penalty_local += 1.0
        per_100_chars = (num_err / max(1, _len_chars)) * 100.0
        base_local = 100.0 - (per_100_chars * 5.0) - (severity_penalty_local * 2.0)
        return clamp(base_local)

    len_chars = len(text)

    # Try cloud Public API first (no Java required)
    try:
        tool = language_tool_python.LanguageToolPublicAPI("en-US")
        matches = tool.check(text)
        score = _score_from_matches(matches, len_chars)
        return score, len(matches), len_chars, matches
    except Exception:
        # Fallback: try local server-backed tool if available (requires Java)
        try:
            tool = language_tool_python.LanguageTool("en-US")
            matches = tool.check(text)
            score = _score_from_matches(matches, len_chars)
            return score, len(matches), len_chars, matches
        except Exception:
            # Last-resort heuristic when LanguageTool is unavailable (offline/rate-limited)
            words = text.split()
            num_words = len(words)
            num_sent_like = max(1, text.count('.') + text.count('!') + text.count('?'))
            avg_words_per_sentence = num_words / max(1, num_sent_like)
            penalties = 0.0
            if "  " in text:
                penalties += 1.0
            if not text.strip().endswith(('.', '!', '?')):
                penalties += 1.5
            sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
            for s in sentences:
                if s and not s[0].isupper():
                    penalties += 0.5
            if num_words < 5:
                penalties += 1.0
            if avg_words_per_sentence > 30:
                penalties += 1.0
            score_est = clamp(85.0 - penalties * 10.0)
            return score_est, 0, len_chars, []


def suggest_corrections(text: str, matches: list) -> str:
    """Return a corrected version of text using LanguageTool matches when available.
    If matches are empty, attempt a best-effort correction using the tool directly.
    Falls back to the original text on any error.
    """
    original = (text or "")
    if not original:
        return original

    def _aux_verb_fix(s: str) -> str:
        """Insert likely-missing auxiliary for continuous aspect (simple heuristic)."""
        def insert_aux(match):
            subj = match.group(1)
            verb_ing = match.group(2)
            aux = {
                "I": "am",
                "You": "are",
                "We": "are",
                "They": "are",
                "He": "is",
                "She": "is",
                "It": "is",
            }.get(subj, "is")
            return f"{subj} {aux} {verb_ing}"

        s = re.sub(r"\b(I|You|We|They|He|She|It)\s+(?!am\b|are\b|is\b)([a-zA-Z]+ing)\b", insert_aux, s)
        return s

    # Prefer the tool's holistic correction first
    corrected_global = None
    try:
        try:
            tool = language_tool_python.LanguageToolPublicAPI("en-US")
            corrected_global = tool.correct(original)
        except Exception:
            tool = language_tool_python.LanguageTool("en-US")
            corrected_global = tool.correct(original)
    except Exception:
        corrected_global = None

    if corrected_global and corrected_global.strip() and corrected_global != original:
        return _aux_verb_fix(corrected_global)

    # If the global correction is unavailable or unchanged, try match-wise replacement
    try:
        if matches:
            try:
                sortable = []
                for m in matches:
                    off = getattr(m, "offset", None)
                    length = getattr(m, "errorLength", None)
                    if length is None:
                        length = getattr(m, "length", None)
                    reps = getattr(m, "replacements", []) or []
                    rep_txt = reps[0].get("value") if reps and isinstance(reps[0], dict) else (reps[0] if reps else None)
                    if off is not None and length is not None and rep_txt is not None:
                        sortable.append((int(off), int(length), str(rep_txt)))

                if sortable:
                    corrected = original
                    for off, length, rep in sorted(sortable, key=lambda x: x[0], reverse=True):
                        corrected = corrected[:off] + rep + corrected[off + length:]
                    if corrected != original:
                        return _aux_verb_fix(corrected)
            except Exception:
                pass
    except Exception:
        pass

    # As a final small improvement, try the auxiliary heuristic on the original
    return _aux_verb_fix(original)

## Removed sentence alternatives logic per user request


def generate_advanced_alternatives(text: str, corrected_text: str, max_items: int = 6) -> list:
    """Produce more polished alternatives using simple style heuristics and synonyms.
    This does not rely on external services; it uses pattern-based rewrites.
    """
    s = (corrected_text or text or "").strip()
    if not s:
        return []

    # Simple sentence split without external dependencies
    sentences = [seg.strip() for seg in re.split(r"(?<=[.!?])\s+", s) if seg.strip()]
    alts = []

    # Lightweight synonym map for common adjectives
    synonym_map = {
        "happy": ["delighted", "thrilled", "elated", "overjoyed", "wonderful"],
        "sad": ["downcast", "sorrowful", "melancholic"],
        "good": ["excellent", "superb", "outstanding"],
        "bad": ["poor", "subpar", "unfavorable"],
        "tired": ["exhausted", "weary", "drained"],
        "angry": ["irate", "furious", "indignant"],
        "nice": ["pleasant", "lovely", "charming"],
        "big": ["large", "substantial", "considerable"],
        "small": ["tiny", "compact", "modest"],
    }

    def boost_degree(phrase: str) -> str:
        # very -> extremely; really -> truly
        phrase = re.sub(r"\bvery\b", "extremely", phrase, flags=re.IGNORECASE)
        phrase = re.sub(r"\breally\b", "truly", phrase, flags=re.IGNORECASE)
        return phrase

    for sent in sentences:
        base = sent
        # Normalize contractions for pattern matching
        norm = re.sub(r"\bI'm\b", "I am", base, flags=re.IGNORECASE)
        norm = re.sub(r"\bYou're\b", "You are", norm, flags=re.IGNORECASE)
        norm = re.sub(r"\bWe're\b", "We are", norm, flags=re.IGNORECASE)
        norm = re.sub(r"\bThey're\b", "They are", norm, flags=re.IGNORECASE)

        # Pattern: "I am feeling <adj>"
        m = re.search(r"\b(I|We|You|They|He|She|It)\s+(am|are|is)\s+feeling\s+([A-Za-z]+)\b", norm, flags=re.IGNORECASE)
        if m:
            subj, _, adj = m.group(1), m.group(2), m.group(3)
            adj_l = adj.lower()
            # Variant 1: simple present (remove auxiliary and conjugate)
            verb = "feels" if subj.lower() in {"he", "she", "it"} else "feel"
            alts.append(re.sub(r"\b(am|are|is)\s+feeling\s+[A-Za-z]+\b", f"{verb} {adj}", base, flags=re.IGNORECASE))
            # Variant 2: adjective predicate (for I/We/You/They)
            if subj.lower() in {"i", "we", "you", "they"}:
                alts.append(re.sub(r"feeling\s+([A-Za-z]+)", f"{adj}", base, flags=re.IGNORECASE))
            # Variant 3: stronger sentiment
            strong = synonym_map.get(adj_l, [adj])
            if strong:
                best = strong[0]
                alts.append(re.sub(r"feeling\s+[A-Za-z]+", f"feeling {best}", base, flags=re.IGNORECASE))
            # Variant 4: natural expression
            if subj.lower() == "i":
                alts.append("I couldn't be happier." if adj_l == "happy" else f"I truly feel {adj}.")
        else:
            # Generic boosters
            boosted = boost_degree(base)
            if boosted != base:
                alts.append(boosted)

        # Generic: replace simple adjective with a more advanced synonym
        def replace_simple_adj(m2):
            word = m2.group(0)
            wlow = word.lower()
            if wlow in synonym_map:
                return synonym_map[wlow][0]
            return word

        adv_sent = re.sub(r"\b(happy|sad|good|bad|tired|angry|nice|big|small)\b", replace_simple_adj, base, flags=re.IGNORECASE)
        if adv_sent != base:
            alts.append(adv_sent)

        if len(alts) >= max_items:
            break

    # Deduplicate and keep readable casing/punctuation
    seen = set()
    uniq = []
    for a in alts:
        a2 = a.strip()
        if not a2:
            continue
        if a2 not in seen and a2 != s:
            seen.add(a2)
            uniq.append(a2)
        if len(uniq) >= max_items:
            break
    return uniq


def extract_grammar_issues(text: str, matches: list, max_issues: int = 50) -> list:
    """Turn LanguageTool matches into readable issues with context and suggestions."""
    issues = []
    if not text or not matches:
        return issues
    total_len = len(text)
    for m in matches[:max_issues]:
        try:
            off = int(getattr(m, "offset", 0))
            length = getattr(m, "errorLength", None)
            if length is None:
                length = getattr(m, "length", 0)
            length = int(length or 0)
            start = max(0, min(off, total_len))
            end = max(start, min(start + length, total_len))
            before_start = max(0, start - 30)
            after_end = min(total_len, end + 30)
            excerpt_before = text[before_start:start]
            error_text = text[start:end] or text[start:start+1]
            excerpt_after = text[end:after_end]
            message = getattr(m, "message", "")
            rule_id = getattr(getattr(m, "rule", None), "id", "") or getattr(m, "ruleId", "")
            reps = getattr(m, "replacements", []) or []
            suggestions = []
            for r in reps[:3]:
                if isinstance(r, dict) and "value" in r:
                    suggestions.append(str(r["value"]))
                else:
                    suggestions.append(str(r))
            issues.append({
                "message": str(message),
                "rule": str(rule_id),
                "before": excerpt_before,
                "error": error_text,
                "after": excerpt_after,
                "suggestions": suggestions,
            })
        except Exception:
            continue
    return issues


def derive_fluency_pct(speaking_rate_wpm: float, grammar_pct: float) -> float:
    # Combine rate closeness to ideal with grammar clarity
    ideal_low, ideal_high = 110.0, 160.0
    if speaking_rate_wpm <= 0:
        rate_component = 50.0
    else:
        if speaking_rate_wpm < ideal_low:
            rate_component = clamp((speaking_rate_wpm / ideal_low) * 100.0)
        elif speaking_rate_wpm > ideal_high:
            rate_component = clamp((ideal_high / speaking_rate_wpm) * 100.0)
        else:
            rate_component = 100.0
    # Weighted blend with grammar quality
    return clamp(0.6 * rate_component + 0.4 * grammar_pct)


def make_feedback(conf_pct: float, gram_pct: float, flu_pct: float, transcript: str) -> str:
    parts = []
    if conf_pct >= 80:
        parts.append("Excellent pronunciation and steadiness.")
    elif conf_pct >= 60:
        parts.append("Good articulation; try to keep amplitude more steady.")
    else:
        parts.append("Work on speaking more clearly and steadily.")

    if gram_pct >= 85:
        parts.append("Grammar is strong with minimal issues.")
    elif gram_pct >= 65:
        parts.append("Minor grammar mistakes detected — review sentence structure.")
    else:
        parts.append("Noticeable grammar and spelling issues — consider revising basics.")

    if flu_pct >= 80:
        parts.append("Fluency and pacing feel natural.")
    elif flu_pct >= 60:
        parts.append("Good structure; try a slightly more consistent pace.")
    else:
        parts.append("Try to maintain a steady pace and reduce long pauses.")

    if transcript and len(transcript.split()) < 5:
        parts.append("Provide longer input for a more reliable evaluation.")

    return " ".join(parts)


# ----------------------------
# Streamlit App UI
# ----------------------------
st.set_page_config(page_title="VoiceEnglish Analyzer", page_icon="🗣️", layout="wide")

# Global theme & styles
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

:root {
    --bg: #0b1220;
    --panel: #0e1726;
    --border: #1f2937;
    --text: #e5e7eb;
    --muted: #cbd5e1;
    --grad-a: #3B82F6; /* blue-500 */
    --grad-b: #10B981; /* emerald-500 */
}

/* Background */
.stApp {
    background: radial-gradient(1200px 600px at 10% 10%, rgba(59,130,246,0.12), rgba(0,0,0,0)) no-repeat,
        linear-gradient(180deg, var(--bg) 0%, var(--bg) 100%);
    font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
    color: var(--text);
}

/* Hero */
.hero {
    border: 1px solid var(--border);
    background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(16,185,129,0.14));
    backdrop-filter: blur(8px);
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.25);
    position: relative;
}
.hero::after {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 16px;
    background: radial-gradient(600px 200px at 0% 0%, rgba(16,185,129,0.12), transparent 60%);
    opacity: 0.6;
    pointer-events: none;
}
.hero h1 { margin: 0; font-weight: 800; letter-spacing: -0.02em; }
.hero p { margin: 6px 0 0 0; color: var(--muted); }

/* Cards */
.card {
    border: 1px solid var(--border);
    background: var(--panel);
    border-radius: 14px;
    padding: 14px 14px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
}

.card h3 {
    margin-top: 0;
}

/* Buttons */
div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, var(--grad-a), var(--grad-b));
    color: var(--bg);
    font-weight: 700;
    border: 0;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    transition: transform .05s ease, filter .15s ease;
}
div.stButton > button:hover { filter: brightness(1.06); }
div.stButton > button:active { transform: translateY(1px); }

/* Inputs */
textarea, .stTextArea textarea, .stTextInput input {
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    background: #0b1526 !important;
    color: var(--text) !important;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 10px;
}

/* Tabs */
div[data-baseweb="tab-list"] > div {
    border-bottom: 1px solid var(--border);
}
div[role="tab"] {
    font-weight: 700;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg);
    border-right: 1px solid var(--border);
}

/* Progress Bars */
.pretty-bar { width: 100%; background: #0b1526; border: 1px solid var(--border); border-radius: 999px; height: 12px; position: relative; overflow: hidden; }
.pretty-bar > span { position: absolute; inset: 0; width: var(--w); background: linear-gradient(90deg, var(--grad-a), var(--grad-b)); box-shadow: 0 0 16px rgba(16,185,129,0.25); }
.pretty-bar-label { display: flex; justify-content: space-between; color: var(--muted); font-size: 0.88rem; margin-bottom: 6px; }
</style>
""",
    unsafe_allow_html=True,
)

# Hero header
st.markdown(
    """
<div class="hero">
    <h1>🗣️ VoiceEnglish Analyzer</h1>
    <p>Analyze your spoken or written English for confidence, grammar, and fluency.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Options")
    mode = st.radio("Choose input mode", options=["Voice", "Text"], horizontal=True)
    st.markdown("---")
    st.write("Tip: Upload clear audio or provide well-formed sentences for best results.")


col_left, col_right = st.columns([1, 1])

overall_score: Optional[float] = None
grammar_score: Optional[float] = None
confidence_score: Optional[float] = None
fluency_pct: Optional[float] = None
transcript_text: str = ""
voice_metrics: Dict[str, float] = {}

if mode == "Voice":
    with col_left:
        st.subheader("🎙️ Provide Voice Input")
        # Option A: Upload audio file
        upload = st.file_uploader(
            "Upload audio (wav, mp3, m4a)", type=["wav", "mp3", "m4a", "ogg", "flac"]
        )
        st.caption("For best results, upload a short, clear clip (10–60 seconds).")

        # In-browser recording (if component available)
        recorded_audio = None
        record_analyze_btn = False
        try:
            from audio_recorder_streamlit import audio_recorder  # type: ignore
            st.caption("Record directly in the browser and analyze instantly.")
            recorded_audio = audio_recorder(text="Start / Stop Recording", pause_threshold=2.0)
            if recorded_audio:
                st.audio(recorded_audio, format="audio/wav")
                record_analyze_btn = st.button("Record and Analyze", type="primary")
        except Exception:
            st.info("To enable in-browser recording, install `audio_recorder_streamlit`.")

        analyze_btn = st.button("Analyze Voice", type="primary")

    if analyze_btn or record_analyze_btn:
        if not upload and not recorded_audio:
            st.warning("Please upload or record audio first.")
        else:
            # Convert input to wav bytes with robust decoding
            wav_bytes = b""
            try:
                if recorded_audio is not None and len(recorded_audio) > 0:
                        wav_bytes = load_audio_and_to_wav_bytes(recorded_audio, filename_suffix=".wav")
                elif upload is not None:
                        # Use uploaded filename extension to aid decoding (e.g., .mp3, .m4a)
                        ext = os.path.splitext(getattr(upload, "name", "audio.tmp"))[1] or ".tmp"
                        wav_bytes = load_audio_and_to_wav_bytes(upload.read(), filename_suffix=ext)
                else:
                        wav_bytes = b""
            except Exception as e:
                st.error(f"Could not read audio. Try uploading a different format. ({e})")
                wav_bytes = b""

            if wav_bytes:
                # Play back
                st.audio(wav_bytes, format="audio/wav")

                # 1) Transcribe first
                transcript_text, api_conf = transcribe_with_speech_recognition(wav_bytes)
                if not transcript_text:
                    st.error("Could not transcribe audio.")
                    st.info("Tips: (1) Ensure internet for Google Web Speech, or (2) Install `pocketsphinx` for offline fallback, and (3) Prefer clear speech in WAV/FLAC or install FFmpeg for MP3/M4A.")
                if api_conf is not None:
                    # Blend API confidence lightly
                    confidence_score = clamp(0.8 * confidence_score + 20.0 * api_conf)

                # 2) Only analyze grammar if we have transcript text
                grammar_score, nerr, lchars, matches = (0.0, 0, 0, [])
                corrected_text = ""
                advanced_alts = []
                issues = []
                if transcript_text and transcript_text.strip():
                    grammar_score, nerr, lchars, matches = grammar_analysis(transcript_text)
                    corrected_text = suggest_corrections(transcript_text, matches)
                    advanced_alts = generate_advanced_alternatives(transcript_text, corrected_text)
                    issues = extract_grammar_issues(transcript_text, matches)

                # 3) Load waveform and compute voice metrics after transcription completes
                y = np.array([])
                sr_hz = 16000
                tmp_wav_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                        tmp_wav.write(wav_bytes)
                        tmp_wav_path = tmp_wav.name
                    y, sr_hz = librosa.load(tmp_wav_path, sr=16000, mono=True)
                except Exception as e:
                    st.warning(f"Could not fully analyze audio waveform. Proceeding with transcript only. ({e})")
                finally:
                    if tmp_wav_path and os.path.exists(tmp_wav_path):
                        try:
                            os.remove(tmp_wav_path)
                        except Exception:
                            pass

                # Guard against silent/too short audio
                if y.size == 0 or np.max(np.abs(y)) < 1e-4:
                    st.warning("Audio appears silent or too short for voice metrics. Skipping voice metrics.")
                    voice_metrics = {"pitch_variation": 0.0, "rms_stability": 0.0, "speaking_rate_wpm": 0.0, "duration_sec": 0.0, "clarity_ratio": 0.0}
                    confidence_score, subs = 50.0, {}
                else:
                    try:
                        voice_metrics = compute_voice_metrics(y, sr_hz)
                        confidence_score, subs = voice_confidence_score(voice_metrics)
                    except Exception as e:
                        st.warning(f"Voice metrics unavailable due to audio characteristics. ({e})")
                        voice_metrics = {"pitch_variation": 0.0, "rms_stability": 0.0, "speaking_rate_wpm": 0.0, "duration_sec": float(len(y)/max(1,sr_hz)), "clarity_ratio": 0.0}
                        confidence_score, subs = 50.0, {}

                # Fluency
                fluency_pct = derive_fluency_pct(voice_metrics.get("speaking_rate_wpm", 0.0), grammar_score)

                # Overall as average of grammar and confidence
                subscores = [x for x in [grammar_score, confidence_score] if x is not None]
                overall_score = float(np.mean(subscores)) if subscores else None

                with col_right:
                    st.subheader("🧾 Results")
                    st.write(f"Transcript: {transcript_text or '—'}")
                    if corrected_text and corrected_text != transcript_text:
                        st.write(f"Corrected: {corrected_text}")
                    if advanced_alts:
                        with st.expander("Advanced alternatives"):
                            for i, a in enumerate(advanced_alts, 1):
                                st.write(f"{i}. {a}")
                    if issues:
                        with st.expander("Grammar issues"):
                            for i, it in enumerate(issues, 1):
                                st.markdown(f"**{i}. {it['message']}**  ")
                                st.code(f"...{it['before']}{it['error']}{it['after']}...")
                                if it["suggestions"]:
                                    st.write("Suggestions: " + ", ".join(it["suggestions"]))
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Confidence", f"{confidence_score:.0f}%")
                    m2.metric("Grammar", f"{grammar_score:.0f}%")
                    m3.metric("Fluency", f"{fluency_pct:.0f}%")

                    if overall_score is not None:
                        st.metric("Overall English Score", f"{overall_score:.0f}%")
                        rating = clamp(overall_score / 10.0, 0, 10)
                        st.subheader(f"⭐ Rating: {rating:.1f}/10")

                        # Pretty progress bars
                        conf_int = int(confidence_score or 0)
                        gram_int = int(grammar_score or 0)
                        flu_int = int(fluency_pct or 0)
                        st.markdown(
                            f"""
                            <div class="card">
                                <h3>Quality Breakdown</h3>
                                <div class="pretty-bar-label"><span>Confidence</span><span>{conf_int}%</span></div>
                                <div class="pretty-bar" style="--w:{conf_int}%"><span></span></div>
                                <div style="height:10px"></div>
                                <div class="pretty-bar-label"><span>Grammar</span><span>{gram_int}%</span></div>
                                <div class="pretty-bar" style="--w:{gram_int}%"><span></span></div>
                                <div style="height:10px"></div>
                                <div class="pretty-bar-label"><span>Fluency</span><span>{flu_int}%</span></div>
                                <div class="pretty-bar" style="--w:{flu_int}%"><span></span></div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Pie chart
                    labels = ["Confidence", "Grammar", "Fluency"]
                    values = [confidence_score or 0, grammar_score or 0, fluency_pct or 0]
                    if PLOTLY_AVAILABLE:
                        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.35)])
                        fig.update_traces(textinfo="percent+label")
                        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig_m, ax_m = plt.subplots(figsize=(4, 4))
                        ax_m.pie(values, labels=labels, autopct="%1.0f%%", startangle=90)
                        ax_m.axis("equal")
                        st.pyplot(fig_m, use_container_width=True)

                    # Feedback
                    fb = make_feedback(confidence_score or 0.0, grammar_score or 0.0, fluency_pct or 0.0, transcript_text)
                    st.info(f"💡 Feedback: {fb}")

else:
    with col_left:
        st.subheader("⌨️ Enter Text")
        user_text = st.text_area("Type or paste your English text here", height=200, placeholder="Write a paragraph for analysis…")
        analyze_text_btn = st.button("Analyze Text", type="primary")

    if analyze_text_btn:
        grammar_score, nerr, lchars, matches = grammar_analysis(user_text)
        corrected_text = suggest_corrections(user_text, matches)
        advanced_alts = generate_advanced_alternatives(user_text, corrected_text)
        issues = extract_grammar_issues(user_text, matches)
        confidence_score = None
        fluency_pct = derive_fluency_pct(140.0, grammar_score)  # assume typical reading rate
        overall_score = grammar_score  # when only text, overall == grammar

        with col_right:
            st.subheader("🧾 Results")
            st.write(f"Text length: {lchars} chars, {len(user_text.split())} words")
            if corrected_text and corrected_text != user_text:
                st.write(f"Corrected: {corrected_text}")
            if advanced_alts:
                with st.expander("Advanced alternatives"):
                    for i, a in enumerate(advanced_alts, 1):
                        st.write(f"{i}. {a}")
            if issues:
                with st.expander("Grammar issues"):
                    for i, it in enumerate(issues, 1):
                        st.markdown(f"**{i}. {it['message']}**  ")
                        st.code(f"...{it['before']}{it['error']}{it['after']}...")
                        if it["suggestions"]:
                            st.write("Suggestions: " + ", ".join(it["suggestions"]))
            m1, m2 = st.columns(2)
            m1.metric("Grammar", f"{grammar_score:.0f}%")
            m2.metric("Fluency", f"{fluency_pct:.0f}%")
            st.metric("Overall English Score", f"{overall_score:.0f}%")
            rating = clamp(overall_score / 10.0, 0, 10)
            st.subheader(f"⭐ Rating: {rating:.1f}/10")

            labels = ["Grammar", "Fluency"]
            values = [grammar_score or 0, fluency_pct or 0]
            if PLOTLY_AVAILABLE:
                fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.35)])
                fig.update_traces(textinfo="percent+label")
                fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig_m, ax_m = plt.subplots(figsize=(4, 4))
                ax_m.pie(values, labels=labels, autopct="%1.0f%%", startangle=90)
                ax_m.axis("equal")
                st.pyplot(fig_m, use_container_width=True)

            # Pretty progress bars
            gram_int = int(grammar_score or 0)
            flu_int = int(fluency_pct or 0)
            st.markdown(
                f"""
                <div class="card">
                    <h3>Quality Breakdown</h3>
                    <div class="pretty-bar-label"><span>Grammar</span><span>{gram_int}%</span></div>
                    <div class="pretty-bar" style="--w:{gram_int}%"><span></span></div>
                    <div style="height:10px"></div>
                    <div class="pretty-bar-label"><span>Fluency</span><span>{flu_int}%</span></div>
                    <div class="pretty-bar" style="--w:{flu_int}%"><span></span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            fb = make_feedback(conf_pct=70.0, gram_pct=grammar_score or 0.0, flu_pct=fluency_pct or 0.0, transcript=user_text)
            st.info(f"💡 Feedback: {fb}")


st.markdown("---")
st.caption("Built with Streamlit, SpeechRecognition, Librosa, and LanguageTool.")
