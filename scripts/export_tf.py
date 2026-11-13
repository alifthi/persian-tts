#!/usr/bin/env python
"""Convert the exported ONNX model to TensorFlow SavedModel (+ TFLite) with onnx2tf, and verify.

Run in the TensorFlow environment (Python <= 3.12):
    .venv-tf/bin/python scripts/export_tf.py --onnx exports/onnx/model.onnx --out exports/tf

TFLite/TF graphs work best with a static text length: inputs are padded to --max-len tokens
(input_lengths carries the true length, so padding does not change the audio).
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pad_ids(ids, n):
    out = np.zeros((1, n), dtype=np.int64)
    out[0, : len(ids)] = ids
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="model exported with export_onnx.py --portable")
    ap.add_argument("--out", default="exports/tf")
    ap.add_argument("--max-len", type=int, default=400, help="static token length (post-blank)")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()

    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")  # onnx2tf calls the onnxsim CLI
    import onnx
    import onnx2tf
    import onnxruntime as ort
    from onnx.tools import update_model_dims

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # fix batch=1 and text length; output time stays dynamic
    m = onnx.load(a.onnx)
    cfg = json.loads(Path(a.onnx + ".json").read_text(encoding="utf-8"))
    n_out = cfg["static_output"]["max_samples"]
    m = update_model_dims.update_inputs_outputs_dims(
        m, {"input": [1, a.max_len], "input_lengths": [1], "scales": [3], "sid": [1]},
        {"output": [1, 1, n_out], "output_lengths": [1]})
    import onnxsim
    m, ok = onnxsim.simplify(m)  # fold all shapes to constants -> unambiguous layouts for onnx2tf
    assert ok, "onnxsim failed"
    static_onnx = out / "model_static.onnx"
    onnx.save(m, str(static_onnx))

    # realistic calibration inputs for onnx2tf's internal shape/accuracy checks (valid speaker id, real text)
    from persian_tts.text import text_to_ids
    calib = text_to_ids("این یک جمله آزمایشی برای تبدیل مدل است.")
    np_dir = out / "calib"
    np_dir.mkdir()
    feeds = dict(input=pad_ids(calib, a.max_len), input_lengths=np.array([len(calib)], dtype=np.int64),
                 scales=np.array([0.667, 1.0, 0.8], dtype=np.float32), sid=np.array([0], dtype=np.int64))
    custom = []
    for k, v in feeds.items():
        np.save(np_dir / f"{k}.npy", v)
        custom.append([k, str(np_dir / f"{k}.npy")])
    onnx2tf.convert(input_onnx_file_path=str(static_onnx), output_folder_path=str(out / "saved_model"),
                    custom_input_op_name_np_data_path=custom,
                    output_signaturedefs=True, keep_shape_absolutely_input_names=["input", "input_lengths", "scales", "sid"],
                    disable_strict_mode=False, non_verbose=True,
                    copy_onnx_input_output_names_to_tflite=True)
    cfg["tf"] = dict(static_input_length=a.max_len, batch=1, note="pad input with 0 to static_input_length; "
                     "trim output to output_lengths samples")
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")

    if a.no_verify:
        return
    import tensorflow as tf

    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    sm = tf.saved_model.load(str(out / "saved_model"))
    fn = sm.signatures["serving_default"]
    for text in ["سلام", "امروز هوا خیلی خوب است و ما به پارک می‌رویم."]:
        ids = text_to_ids(text)
        feed = dict(input=pad_ids(ids, a.max_len), input_lengths=np.array([len(ids)], dtype=np.int64),
                    scales=np.array([0.0, 1.0, 0.8], dtype=np.float32), sid=np.array([0], dtype=np.int64))
        ref, ref_len = sess.run(None, {**feed, "input": np.array([ids], dtype=np.int64)})
        ref = ref.reshape(-1)[: int(ref_len[0])]
        res = fn(**{k: tf.constant(v) for k, v in feed.items()})
        audio = [v for v in res.values() if v.shape.rank == 3][0].numpy().reshape(-1)
        n = int([v for v in res.values() if v.shape.rank == 1][0].numpy()[0])
        assert n == ref.size, f"length mismatch tf={n} onnx={ref.size}"
        err = np.abs(audio[:n] - ref).max()
        print(f"tf check len={len(ids):3d}: samples={n} max_abs_err={err:.2e}")
        assert err < 1e-2, "TensorFlow output deviates from ONNX"
    # TFLite (float32) check
    tfl = sorted((out / "saved_model").glob("*_float32.tflite"))[0]
    interp = tf.lite.Interpreter(model_path=str(tfl))
    interp.allocate_tensors()
    ids = text_to_ids("امروز هوا خیلی خوب است و ما به پارک می‌رویم.")
    feed = dict(input=pad_ids(ids, a.max_len), input_lengths=np.array([len(ids)], dtype=np.int64),
                scales=np.array([0.0, 1.0, 0.8], dtype=np.float32), sid=np.array([0], dtype=np.int64))
    for d in interp.get_input_details():
        key = next(k for k in feed if d["name"].endswith(k) or k in d["name"].split(":")[0].split("_")[-1:] or d["name"] == k)
        interp.set_tensor(d["index"], feed[key].astype(d["dtype"]))
    interp.invoke()
    outs = [interp.get_tensor(d["index"]) for d in interp.get_output_details()]
    audio = next(o for o in outs if o.ndim == 3).reshape(-1)
    ref, ref_len = sess.run(None, {**feed, "input": np.array([ids], dtype=np.int64)})
    n = int(ref_len[0])
    print(f"tflite check: max_abs_err={np.abs(audio[:n] - ref.reshape(-1)[:n]).max():.2e}")
    print("SavedModel:", out / "saved_model", "| TFLite:", sorted(p.name for p in (out / "saved_model").glob("*.tflite")))


if __name__ == "__main__":
    main()
