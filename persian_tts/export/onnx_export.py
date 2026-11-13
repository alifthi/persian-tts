"""Export the generator to ONNX (Piper-compatible I/O) and verify against PyTorch."""
import json
from pathlib import Path

import numpy as np
import torch

from ..config import ModelConfig, AudioConfig
from ..models import SynthesizerTrn
from ..text import SYMBOLS, text_to_ids
from ..text.symbols import BLANK_ID, PAD_ID


class OnnxWrapper(torch.nn.Module):
    def __init__(self, g: SynthesizerTrn, hop: int = 256):
        super().__init__()
        self.g, self.hop = g, hop

    def forward(self, input, input_lengths, scales, sid):  # noqa: A002 (Piper names)
        audio, y_lengths = self.g.infer_onnx(input, input_lengths, scales, sid)
        if self.g.static_max_frames:  # fixed-size output + valid sample count
            return audio, y_lengths * self.hop
        return audio


def load_generator(ckpt_path: str, use_ema: bool = True):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    mcfg = ModelConfig(**cfg["model"])
    acfg = AudioConfig(**cfg["audio"])
    assert ck["symbols"] == SYMBOLS, "checkpoint symbol table differs from code"
    g = SynthesizerTrn(len(SYMBOLS), acfg.n_fft // 2 + 1, 32, len(ck["speakers"]), mcfg)
    g.load_state_dict(ck["ema"] if use_ema and "ema" in ck else ck["net_g"])
    g.eval()
    g.remove_weight_norm()
    return g, ck, acfg


def export(ckpt_path: str, out_dir: str, opset: int = 17, verify: bool = True, portable: bool = False,
           max_frames: int = 2600) -> Path:
    """portable=True: RNG-free noise + static output length (max_frames*256 samples), for TensorFlow/TFLite."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g, ck, acfg = load_generator(ckpt_path)
    g.portable_noise = portable  # TensorFlow has no RandomNormalLike
    g.static_max_frames = max_frames if portable else None
    model = OnnxWrapper(g, acfg.hop_length).eval()

    ids = torch.LongTensor([text_to_ids("سلام، این یک آزمایش است.")])
    dummy = (ids, torch.LongTensor([ids.shape[1]]), torch.FloatTensor([0.667, 1.0, 0.8]), torch.LongTensor([0]))
    onnx_path = out / "model.onnx"
    outputs = ["output", "output_lengths"] if portable else ["output"]
    dyn = {"input": {0: "batch", 1: "phonemes"}, "input_lengths": {0: "batch"}, "sid": {0: "batch"},
           "output": {0: "batch"} if portable else {0: "batch", 2: "time"}}
    torch.onnx.export(model, dummy, str(onnx_path), opset_version=opset, dynamo=False,
                      input_names=["input", "input_lengths", "scales", "sid"], output_names=outputs, dynamic_axes=dyn)
    try:
        import onnxsim
        import onnx
        m, ok = onnxsim.simplify(onnx.load(str(onnx_path)))
        if ok:
            onnx.save(m, str(onnx_path))
    except ImportError:
        pass

    # Piper-style runtime config: everything needed to tokenize and run the model
    (out / "model.onnx.json").write_text(json.dumps(dict(
        audio=dict(sample_rate=acfg.sample_rate), num_symbols=len(SYMBOLS), num_speakers=len(ck["speakers"]),
        speaker_id_map={s: i for i, s in enumerate(ck["speakers"])},
        symbols=SYMBOLS, pad_id=PAD_ID, blank_id=BLANK_ID, add_blank=True, phoneme_type="text",
        language=dict(code="fa_IR"), inference=dict(noise_scale=0.667, length_scale=1.0, noise_w=0.8),
        static_output=dict(max_samples=max_frames * acfg.hop_length) if portable else None,
    ), ensure_ascii=False, indent=1), encoding="utf-8")

    if verify:
        verify_onnx(onnx_path, g)
    return onnx_path


def verify_onnx(onnx_path, g, atol=1e-3):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for text in ["سلام", "امروز هوا خیلی خوب است و ما ساعت ۵ به پارک می‌رویم."]:
        ids = torch.LongTensor([text_to_ids(text)])
        scales = torch.FloatTensor([0.0, 1.0, 0.8])  # noise off -> exact graph comparison
        sid = torch.LongTensor([0])
        with torch.no_grad():
            ref, _ = g.infer_onnx(ids, torch.LongTensor([ids.shape[1]]), scales, sid)
        res = sess.run(None, {"input": ids.numpy(), "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
                              "scales": scales.numpy(), "sid": sid.numpy()})
        out = res[0]
        if len(res) > 1:  # static output: trim to valid length
            out = out[..., : int(res[1][0])]
            ref = ref[..., : out.shape[-1]]
        assert out.shape == tuple(ref.shape), f"shape mismatch {out.shape} vs {tuple(ref.shape)}"
        err = np.abs(out - ref.numpy()).max()
        print(f"onnx check len={ids.shape[1]:3d}: samples={out.shape[-1]} max_abs_err={err:.2e}")
        assert err < atol, "ONNX output deviates from PyTorch"
