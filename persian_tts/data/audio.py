"""Audio preprocessing shared by data preparation and training."""
import numpy as np


def preprocess_audio(audio: np.ndarray, sr: int, target_sr: int, trim_db: float = 40.0,
                     target_rms_db: float = -20.0, margin_s: float = 0.05) -> np.ndarray:
    import librosa

    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
    # trim leading/trailing silence, keep a small margin
    _, (s, e) = librosa.effects.trim(audio, top_db=trim_db, frame_length=1024, hop_length=256)
    m = int(margin_s * target_sr)
    audio = audio[max(0, s - m): min(len(audio), e + m)]
    audio = audio - audio.mean()  # DC
    rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
    audio = audio * (10 ** (target_rms_db / 20) / rms)
    peak = np.abs(audio).max() + 1e-9
    if peak > 0.97:
        audio = audio * (0.97 / peak)
    return audio.astype(np.float32)
