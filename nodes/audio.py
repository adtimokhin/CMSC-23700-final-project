import numpy as np
from nodes.base import Node


class AudioInputNode(Node):
    """Loads an MP3/WAV file and sets up timing information."""

    def __init__(self, filepath: str, fps: int = 24, start: float = None, end: float = None, name: str = None):
        """
        filepath: path to MP3/WAV file
        fps: frames per second
        start: start time in seconds (None = beginning of file)
        end: end time in seconds (None = end of file)
        """
        super().__init__(name)
        self.filepath = filepath
        self.fps = fps
        self.start = start
        self.end = end

    def process(self, data: dict) -> dict:
        import librosa

        y, sr = librosa.load(self.filepath, sr=None)

        # Trim to [start, end] segment
        if self.start is not None:
            y = y[int(self.start * sr):]
        if self.end is not None:
            end_sample = int(self.end * sr)
            if self.start is not None:
                end_sample -= int(self.start * sr)
            y = y[:end_sample]

        duration = len(y) / sr
        n_frames = int(duration * self.fps)

        data["audio_raw"] = y
        data["audio_sr"] = sr
        data["fps"] = self.fps
        data["duration"] = duration
        data["n_frames"] = n_frames
        data["time"] = np.linspace(0, duration, n_frames)

        return data


class AudioAnalysisNode(Node):
    """
    Extracts audio features: volume envelope, frequency band energies, onset strength.
    All outputs are per-frame arrays normalized to [0, 1].
    """

    def __init__(self, name: str = None):
        super().__init__(name)

    def validate(self, data: dict):
        for key in ("audio_raw", "audio_sr", "n_frames"):
            if key not in data:
                raise ValueError(
                    f"AudioAnalysisNode requires '{key}' in data. "
                    "Add AudioInputNode before this node."
                )

    def process(self, data: dict) -> dict:
        import librosa

        y = data["audio_raw"]
        sr = data["audio_sr"]
        n_frames = data["n_frames"]

        hop_length = max(1, len(y) // n_frames)

        # RMS volume envelope
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        # Mel spectrogram for frequency bands
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, hop_length=hop_length, n_mels=128
        )
        S_db = librosa.power_to_db(S)

        # Split into bass (~0-300Hz), mid (~300-2kHz), high (~2k+)
        bass = np.mean(S_db[:10, :], axis=0)
        mid = np.mean(S_db[10:60, :], axis=0)
        high = np.mean(S_db[60:, :], axis=0)

        # Onset detection for beat reactivity
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

        data["audio_volume"] = _resample_and_normalize(rms, n_frames)
        data["audio_bass"] = _resample_and_normalize(bass, n_frames)
        data["audio_mid"] = _resample_and_normalize(mid, n_frames)
        data["audio_high"] = _resample_and_normalize(high, n_frames)
        data["audio_onset"] = _resample_and_normalize(onset_env, n_frames)

        return data


def _resample_and_normalize(arr: np.ndarray, n_frames: int) -> np.ndarray:
    """Resample an array to n_frames length and normalize to [0, 1]."""
    resampled = np.interp(
        np.linspace(0, len(arr) - 1, n_frames),
        np.arange(len(arr)),
        arr,
    )
    mn, mx = resampled.min(), resampled.max()
    if mx - mn > 1e-8:
        return (resampled - mn) / (mx - mn)
    return np.zeros(n_frames)
