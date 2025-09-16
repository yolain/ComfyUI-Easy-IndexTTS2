import json
import numpy as np
import torch
from typing import Optional, Tuple

from .model_loader import IndexTTS2Loader
from .utils import save_temp_wav


class IndexTTS2Engine:
    """
    Thin wrapper calling 原项目代码/index-tts/indextts/infer_v2.IndexTTS2.infer.
    It converts ComfyUI audio inputs to temp WAV files, forwards parameters,
    and converts returned audio back to numpy.
    """

    def __init__(self, loader: Optional[IndexTTS2Loader] = None):
        self.loader = loader or IndexTTS2Loader()
        self.tts = None

    def unload_model(self):
        if self.loader and hasattr(self.loader, "_cache"):
            self.loader._cache.clear()
        self.tts = None
        torch.cuda.empty_cache()

    def _estimate_tokens_from_duration(self, seconds: float) -> int:
        # Rough heuristic: 1s speech ~= 75 mel tokens (tunable). Clamp to 50..3000
        if seconds <= 0:
            return 1500
        est = int(seconds * 75)
        return max(50, min(3000, est))

    def generate(
        self,
        text: str,
        reference_audio: Optional[Tuple[np.ndarray, int]] = None,
        style_text: Optional[str] = None,
        style_audio: Optional[Tuple[np.ndarray, int]] = None,
        mode: str = "Auto",  # kept for backward-compat, not used
        duration_sec: Optional[float] = None,  # ignored (HF demo doesn't use it)
        token_count: Optional[int] = None,     # ignored (HF demo doesn't use it)
        # Advanced generation controls
        do_sample: bool = False,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 30,
        num_beams: int = 3,
        repetition_penalty: float = 10.0,
        length_penalty: float = 0.0,
        max_mel_tokens: int = 1500,
        max_tokens_per_sentence: int = 120,
        # Emotion controls
        emotion_control_method: Optional[str] = None,
        emo_text: Optional[str] = None,
        emo_ref_audio: Optional[Tuple[np.ndarray, int]] = None,
        emo_vector: Optional[list] = None,
        emo_weight: float = 0.8,
        seed: int = 0,
        use_qwen: bool = False,
        verbose: bool = False,
        return_subtitles: bool = True,
        use_random: bool = False,
    ) -> Tuple[int, np.ndarray, Optional[str]]:
        # Ensure model is available
        tts = self.tts

        if reference_audio is None:
            raise ValueError("reference_audio is required for IndexTTS2")
        spk_wav_path = save_temp_wav(reference_audio)

        emo_wav_path = None
        if emo_ref_audio is not None:
            emo_wav_path = save_temp_wav(emo_ref_audio)
        elif style_audio is not None:
            # backward compat: treat legacy style_audio as emo_ref_audio
            emo_wav_path = save_temp_wav(style_audio)

        # Follow HF demo strictly: do not remap length by duration or tokens here.
        # Use the given max_mel_tokens (default 1815 in our nodes) directly.
        _max_mel_tokens = int(max_mel_tokens) if max_mel_tokens else 1500

        # Generation kwargs aligned with infer_v2
        gen_kwargs = dict(
            do_sample=bool(do_sample),
            top_p=float(top_p),
            top_k=int(top_k),
            temperature=float(temperature),
            length_penalty=float(length_penalty),
            num_beams=int(num_beams),
            repetition_penalty=float(repetition_penalty),
            max_mel_tokens=int(_max_mel_tokens),
        )

        # Emotion control selection
        # Emotion selection priority: emo_ref_audio > emo_vector > emo_text(use_qwen)
        use_emo_text = False
        _emo_text = None
        if emo_wav_path is None and (emo_vector is None or len(emo_vector) == 0):
            if use_qwen and emo_text and str(emo_text).strip():
                use_emo_text = True
                _emo_text = emo_text

        # Call upstream infer; output_path=None returns (sr, wav_int16_numpy_TxC)
        result = tts.infer(
            spk_audio_prompt=spk_wav_path,
            text=text,
            output_path=None,
            emo_audio_prompt=emo_wav_path,
            emo_alpha=float(emo_weight),
            emo_vector=emo_vector if (emo_wav_path is None and emo_vector) else None,
            use_emo_text=bool(use_emo_text),
            emo_text=_emo_text,
            use_random=use_random,
            interval_silence=200,
            verbose=bool(verbose),
            max_text_tokens_per_segment=int(max_tokens_per_sentence) if max_tokens_per_sentence else 120,
            **gen_kwargs,
        )

        if not (isinstance(result, tuple) and len(result) == 2):
            raise RuntimeError(f"Unexpected return from IndexTTS2.infer: {type(result)}")

        sr, wav = result
        # wav is int16 numpy with shape [T, C] (from upstream .T). Convert to mono float32
        wav = np.asarray(wav)
        if wav.ndim == 2:
            # average channels
            wav = wav.mean(axis=1)
        wav = (wav.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

        subtitle = None
        if return_subtitles:
            # Minimal single-span subtitle; Pro node will have multi-seg support later
            duration = len(wav) / float(sr)
            subtitle = json.dumps([
                {"id": "Narrator", "字幕": text, "start": 0.0, "end": round(duration, 2)}
            ], ensure_ascii=False)

        return int(sr), wav, subtitle
