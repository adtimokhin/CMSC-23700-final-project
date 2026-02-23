import numpy as np
from nodes.base import Node

"""
This is the core set of notes that makes the magic work. Currently, just for demonstration purposes,
it is implemented via Librosa, a different library that implements parsing of music.
Hopefully, most of it can be implemented by me before the final project! 
"""

# FIXME: Right now, there is no implementation of FFT. This will be fixed in the future, before final submission. 
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

        # sr=None tells librosa to keep the file's native sample rate instead
        # of resampling to its default 22050 Hz — avoids unnecessary quality loss.
        y, sr = librosa.load(self.filepath, sr=None)

        # Trim to [start, end] segment by converting seconds to sample indices.
        # We subtract start's offset from end_sample because after the first
        # slice y already starts at 0 — the array is re-indexed, not the original.
        if self.start is not None:
            y = y[int(self.start * sr):]
        if self.end is not None:
            end_sample = int(self.end * sr)
            if self.start is not None:
                end_sample -= int(self.start * sr)
            y = y[:end_sample]

        duration = len(y) / sr
        n_frames = int(duration * self.fps)

        # Setting some variables
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

    def process(self, data: dict) -> dict:
        import librosa

        y = data["audio_raw"]
        sr = data["audio_sr"]
        n_frames = data["n_frames"]

        # hop_length controls how many audio samples sit between consecutive
        # analysis frames. Matching it to n_frames gives us roughly one
        # analysis window per render frame, so the arrays align naturally.
        hop_length = max(1, len(y) // n_frames)

        # RMS (root mean square) energy = perceptual loudness per window.
        # [0] because librosa returns shape (1, n_windows).
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        # Mel spectrogram warps the frequency axis to match how human hearing
        # perceives pitch — lower frequencies get more bins than higher ones.
        # n_mels=128 gives us 128 frequency bands across the full spectrum.
        # https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, hop_length=hop_length, n_mels=128
        )
        # Convert power to dB so the values are perceptually uniform rather
        # than exponentially weighted toward louder frequencies.
        S_db = librosa.power_to_db(S)

        # Bins 0-9 cover bass register
        # Bins 10-59 cover mid ranger
        # Bins 60+ cover high frequencies
        bass = np.mean(S_db[:10, :], axis=0)
        mid = np.mean(S_db[10:60, :], axis=0)
        high = np.mean(S_db[60:, :], axis=0)

        # Onset strength measures how abruptly the spectrum changes frame-to-frame.
        # High values = sharp transient (drum hit, note attack). Good for beat sync.
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

        # Resample each feature to exactly n_frames and squash to [0, 1] so
        # downstream nodes can treat every source uniformly as a 0-1 driver.
        data["audio_volume"] = _resample_and_normalize(rms, n_frames)
        data["audio_bass"] = _resample_and_normalize(bass, n_frames)
        data["audio_mid"] = _resample_and_normalize(mid, n_frames)
        data["audio_high"] = _resample_and_normalize(high, n_frames)
        data["audio_onset"] = _resample_and_normalize(onset_env, n_frames)

        return data


def _resample_and_normalize(arr: np.ndarray, n_frames: int) -> np.ndarray:
    """Resample an array to n_frames length and normalize to [0, 1]."""
    # np.interp does linear interpolation: we evaluate the original array at
    # n_frames evenly-spaced fractional indices, stretching or shrinking it.
    resampled = np.interp(
        np.linspace(0, len(arr) - 1, n_frames),
        np.arange(len(arr)),
        arr,
    )
    min_val, max_val = resampled.min(), resampled.max()

    if max_val - min_val > 1e-8: # This difference is so small, we can just turn evrything into zeros
        return (resampled - min_val) / (max_val - min_val)
    return np.zeros(n_frames)
