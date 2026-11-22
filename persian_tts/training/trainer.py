"""GAN trainer for VITS: DDP, AMP, EMA, checkpointing, TensorBoard."""
import copy
import json
import math
import os
import signal
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..config import ExperimentConfig, to_dict
from ..data.dataset import BucketBatchSampler, TTSDataset, collate
from ..data.mel import mel_spectrogram, spec_to_mel
from ..models import MultiPeriodDiscriminator, SynthesizerTrn
from ..models import commons
from ..text import get_frontend
from .losses import discriminator_loss, feature_loss, generator_loss


def _unwrap(m):
    return m.module if isinstance(m, DDP) else m


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(_unwrap(model)).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, p in zip(self.shadow.parameters(), _unwrap(model).parameters()):
            e.lerp_(p.detach(), 1 - self.decay)
        for e, b in zip(self.shadow.buffers(), _unwrap(model).buffers()):
            e.copy_(b)


class Trainer:
    def __init__(self, cfg: ExperimentConfig, resume: bool = True):
        self.cfg = cfg
        self.ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
        if self.ddp:
            dist.init_process_group("nccl")
            self.rank, self.world = dist.get_rank(), dist.get_world_size()
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        else:
            self.rank, self.world = 0, 1
        self.device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        self.is_main = self.rank == 0
        torch.manual_seed(cfg.train.seed + self.rank)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.run_dir = Path(cfg.out_dir) / cfg.name
        self.ckpt_dir = self.run_dir / "checkpoints"
        if self.is_main:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "config.json").write_text(json.dumps(to_dict(cfg), indent=1))
            self.writer = SummaryWriter(self.run_dir / "tb")

        a, d = cfg.audio, cfg.data
        self.speakers = json.loads(Path(d.speakers).read_text(encoding="utf-8"))
        if self.is_main:
            (self.run_dir / "speakers.json").write_text(json.dumps(self.speakers, ensure_ascii=False, indent=1))
        self.symbols, _ = get_frontend(d.text_frontend)
        self.train_ds = TTSDataset(d.train_manifest, d.root, self.speakers, a, d.add_blank, frontend=d.text_frontend)
        self.val_ds = TTSDataset(d.val_manifest, d.root, self.speakers, a, d.add_blank, frontend=d.text_frontend)
        self.sampler = BucketBatchSampler(self.train_ds.lengths, d.max_frames_per_batch, d.max_batch_size,
                                          self.world, self.rank, shuffle=True, seed=cfg.train.seed)
        self.train_dl = DataLoader(self.train_ds, batch_sampler=self.sampler, collate_fn=collate,
                                   num_workers=d.num_workers, pin_memory=True,
                                   persistent_workers=d.num_workers > 0, prefetch_factor=4 if d.num_workers else None)

        seg_frames = cfg.train.segment_size // a.hop_length
        self.net_g = SynthesizerTrn(len(self.symbols), a.n_fft // 2 + 1, seg_frames, len(self.speakers), cfg.model).to(self.device)
        self.net_d = MultiPeriodDiscriminator().to(self.device)
        t = cfg.train
        dec_ids = {id(p) for p in self.net_g.dec.parameters()}
        groups = [dict(params=[p for p in self.net_g.parameters() if id(p) not in dec_ids]),
                  dict(params=list(self.net_g.dec.parameters()))]  # group 1 = decoder (vocoder)
        self.optim_g = torch.optim.AdamW(groups, t.learning_rate, betas=tuple(t.betas), eps=t.eps)
        self.optim_d = torch.optim.AdamW(self.net_d.parameters(), t.learning_rate_d, betas=tuple(t.betas_d), eps=t.eps)
        steps_per_epoch = max(1, len(self.sampler))
        gamma = t.lr_decay ** (1.0 / steps_per_epoch)

        def anneal(s):
            """Cosine multiplier in [anneal_final_scale, 1]; 1.0 when annealing is disabled."""
            if t.anneal_end_step <= 0 or s <= t.anneal_start_step:
                return 1.0
            if s >= t.anneal_end_step:
                return t.anneal_final_scale
            frac = (s - t.anneal_start_step) / max(1, t.anneal_end_step - t.anneal_start_step)
            return t.anneal_final_scale + (1.0 - t.anneal_final_scale) * 0.5 * (1.0 + math.cos(math.pi * frac))

        lr_lambda = lambda s: min(1.0, (s + 1) / max(1, t.warmup_steps)) * gamma ** s * anneal(s)  # noqa: E731
        # decoder lr is 0 during the freeze window (AdamW decay is lr-scaled, so weights stay exact)
        dec_lambda = lambda s: 0.0 if s < t.freeze_decoder_steps else lr_lambda(s)  # noqa: E731
        self.sched_g = torch.optim.lr_scheduler.LambdaLR(self.optim_g, [lr_lambda, dec_lambda])
        self.sched_d = torch.optim.lr_scheduler.LambdaLR(self.optim_d, lr_lambda)

        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(t.precision)
        if self.amp_dtype is torch.bfloat16 and self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.float16
        self.scaler = torch.amp.GradScaler(enabled=self.amp_dtype is torch.float16)

        self.step, self.epoch, self.best_val = 0, 0, float("inf")
        self._ema_state = None
        if t.init_from:
            self._warm_start(t.init_from)
        if resume:
            self._maybe_resume()
        self.ema = EMA(self.net_g, t.ema_decay)
        if self._ema_state is not None:
            self.ema.shadow.load_state_dict(self._ema_state)
        if self.ddp:
            self.net_g = DDP(self.net_g, device_ids=[self.device.index])
            self.net_d = DDP(self.net_d, device_ids=[self.device.index])
        if t.compile:
            self.net_d = torch.compile(self.net_d)
        self._stop = False
        signal.signal(signal.SIGTERM, self._on_sigterm)  # preemptible clusters: save then exit

    # ------------------------------------------------------------------ checkpoints
    def _on_sigterm(self, *_):
        self._stop = True

    def _warm_start(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("symbols") is not None and list(ck["symbols"]) != list(self.symbols):
            raise ValueError(f"{path} was built for a different text frontend than data.text_frontend="
                             f"{self.cfg.data.text_frontend!r}; rebuild it with convert_piper_checkpoint.py --frontend")
        for name, net, key in (("G", self.net_g, "ema" if "ema" in ck else "net_g"), ("D", self.net_d, "net_d")):
            sd = ck.get(key)
            if sd is None:
                continue
            own = net.state_dict()
            ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
            net.load_state_dict(ok, strict=False)
            if self.is_main:
                missing = [k for k in own if k not in ok]
                print(f"warm start {name} from {path}: loaded {len(ok)}/{len(own)} tensors; "
                      f"newly initialized: {sorted({m.split('.')[0] + '.' + m.split('.')[1] for m in missing})}")

    def _maybe_resume(self):
        latest = self.ckpt_dir / "latest.pt"
        if not latest.exists():
            return
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        self.net_g.load_state_dict(ck["net_g"])
        self.net_d.load_state_dict(ck["net_d"])
        self.optim_g.load_state_dict(ck["optim_g"])
        self.optim_d.load_state_dict(ck["optim_d"])
        self.sched_g.load_state_dict(ck["sched_g"])
        self.sched_d.load_state_dict(ck["sched_d"])
        self.scaler.load_state_dict(ck["scaler"])
        self.step, self.epoch, self.best_val = ck["step"], ck["epoch"], ck.get("best_val", float("inf"))
        self._ema_state = ck["ema"]
        if self.is_main:
            print(f"resumed from step {self.step}")

    def save(self, tag=None):
        if not self.is_main:
            return
        ck = dict(net_g=_unwrap(self.net_g).state_dict(), net_d=_unwrap(self.net_d).state_dict(),
                  ema=self.ema.shadow.state_dict(), optim_g=self.optim_g.state_dict(),
                  optim_d=self.optim_d.state_dict(), sched_g=self.sched_g.state_dict(),
                  sched_d=self.sched_d.state_dict(), scaler=self.scaler.state_dict(),
                  step=self.step, epoch=self.epoch, best_val=self.best_val, config=to_dict(self.cfg),
                  speakers=self.speakers, symbols=self.symbols)
        name = tag or f"step_{self.step:08d}.pt"
        tmp = self.ckpt_dir / (name + ".tmp")
        torch.save(ck, tmp)
        os.replace(tmp, self.ckpt_dir / name)  # atomic
        if tag is None:
            (self.ckpt_dir / "latest.pt").unlink(missing_ok=True)
            os.link(self.ckpt_dir / name, self.ckpt_dir / "latest.pt")
            olds = sorted(self.ckpt_dir.glob("step_*.pt"))[: -self.cfg.train.keep_last]
            for o in olds:
                o.unlink()

    # ------------------------------------------------------------------ training
    def _mel(self, spec_or_wav, is_wav):
        a = self.cfg.audio
        if is_wav:
            return mel_spectrogram(spec_or_wav.squeeze(1), a.sample_rate, a.n_fft, a.hop_length, a.win_length,
                                   a.n_mels, a.fmin, a.fmax)
        return spec_to_mel(spec_or_wav, a.sample_rate, a.n_fft, a.n_mels, a.fmin, a.fmax)

    def train_step(self, batch):
        t, a = self.cfg.train, self.cfg.audio
        b = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        ac = torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None)
        seg = t.segment_size // a.hop_length

        with ac:
            y_hat, l_dur, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = self.net_g(
                b["text"], b["text_lengths"], b["spec"], b["spec_lengths"], b["sid"])
            mel = self._mel(b["spec"], False)
            y_mel = commons.slice_segments(mel, ids_slice, seg)
            y = commons.slice_segments(b["wav"], ids_slice * a.hop_length, t.segment_size)
        y_hat_mel = self._mel(y_hat.float(), True)

        # ---- discriminator
        with ac:
            y_d_r, y_d_g, _, _ = self.net_d(y, y_hat.detach())
            loss_d = discriminator_loss(y_d_r, y_d_g)
        self.optim_d.zero_grad(set_to_none=True)
        self.scaler.scale(loss_d).backward()
        self.scaler.unscale_(self.optim_d)
        gn_d = torch.nn.utils.clip_grad_norm_(self.net_d.parameters(), t.grad_clip)
        self.scaler.step(self.optim_d)

        # ---- generator
        with ac:
            _, y_d_g, fmap_r, fmap_g = self.net_d(y, y_hat)
        loss_mel = torch.nn.functional.l1_loss(y_mel.float(), y_hat_mel) * t.c_mel
        loss_kl = commons.kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * t.c_kl
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen = generator_loss(y_d_g)
        loss_g = loss_gen + loss_fm + loss_mel + l_dur + loss_kl
        self.optim_g.zero_grad(set_to_none=True)
        self.scaler.scale(loss_g).backward()
        self.scaler.unscale_(self.optim_g)
        gn_g = torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), t.grad_clip)
        self.scaler.step(self.optim_g)
        self.scaler.update()
        self.sched_g.step()
        self.sched_d.step()
        self.ema.update(self.net_g)
        return dict(loss_d=loss_d, loss_g=loss_g, loss_gen=loss_gen, loss_fm=loss_fm, loss_mel=loss_mel,
                    loss_dur=l_dur, loss_kl=loss_kl, grad_norm_g=gn_g, grad_norm_d=gn_d)

    @torch.no_grad()
    def evaluate(self, n_audio=4):
        """Teacher-forced mel L1 + duration loss on the full val set (EMA weights), and TTS audio samples."""
        g = self.ema.shadow
        if self.device.type == "cuda":
            torch.cuda.empty_cache()  # return cached training blocks before full-length decoding
        dl = DataLoader(self.val_ds, batch_size=self.cfg.data.eval_batch_size, collate_fn=collate, num_workers=2)
        tot_mel, tot_dur, n = 0.0, 0.0, 0
        for i, batch in enumerate(dl):
            try:
                res = self._eval_batch(g, batch, i, n_audio)
            except torch.OutOfMemoryError:  # a long val utterance must not kill a multi-day run
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"[eval] skipped val batch {i}: out of memory")
                continue
            tot_mel, tot_dur, n = tot_mel + res[0], tot_dur + res[1], n + 1
        return dict(val_mel_l1=tot_mel / max(n, 1), val_len_err=tot_dur / max(n, 1))

    def _eval_batch(self, g, batch, i, n_audio):
        """-> (mel L1, relative length error) for one val batch; logs TTS audio for the first one."""
        a = self.cfg.audio
        b = {k: v.to(self.device) for k, v in batch.items()}
        sid = b["sid"]
        gg = g._g(sid)
        x, m_p, logs_p, x_mask = g.enc_p(b["text"], b["text_lengths"])
        z, _, _, y_mask = g.enc_q(b["spec"], b["spec_lengths"], g=gg)
        y_hat = g.dec(z * y_mask, g=gg)
        mel = self._mel(b["spec"], False)
        mel_hat = self._mel(y_hat.squeeze(1).unsqueeze(1), True)[..., : mel.shape[-1]]
        m = y_mask[..., : mel_hat.shape[-1]]
        mel_l1 = (torch.abs(mel[..., : mel_hat.shape[-1]] - mel_hat) * m).sum().item() / (m.sum().item() * a.n_mels)
        # duration error: predicted total length vs. ground truth frames
        logw = g.dp(x, x_mask, g=gg)
        pred_len = torch.ceil(torch.exp(logw) * x_mask).sum([1, 2])
        len_err = (torch.abs(pred_len - b["spec_lengths"]) / b["spec_lengths"]).mean().item()
        if i == 0 and self.is_main:
            for k in range(min(n_audio, b["text"].shape[0])):
                tl = b["text_lengths"][k: k + 1]
                o, yl = g.infer(b["text"][k: k + 1, : tl.item()], tl, sid=sid[k: k + 1] if sid is not None else None)
                self.writer.add_audio(f"tts/{k}", o[0, 0].float().cpu(), self.step, a.sample_rate)
                if self.step == 0 or not getattr(self, "_logged_gt", False):
                    self.writer.add_audio(f"gt/{k}", b["wav"][k, 0, : b["wav_lengths"][k]].cpu(), self.step, a.sample_rate)
            self._logged_gt = True
        return mel_l1, len_err

    def fit(self):
        t = self.cfg.train
        self.net_g.train()
        self.net_d.train()
        t0, acc = time.time(), {}
        while self.step < t.max_steps and not self._stop:
            self.sampler.set_epoch(self.epoch)
            for batch in self.train_dl:
                logs = self.train_step(batch)
                self.step += 1
                for k, v in logs.items():
                    acc[k] = acc.get(k, 0.0) + v.detach().float()  # no host sync per step
                if self.is_main and self.step % t.log_every == 0:
                    dt = time.time() - t0
                    msg = {k: float(v) / t.log_every for k, v in acc.items()}
                    for k, v in msg.items():
                        self.writer.add_scalar(f"train/{k}", v, self.step)
                    self.writer.add_scalar("train/lr", self.sched_g.get_last_lr()[0], self.step)
                    self.writer.add_scalar("perf/steps_per_sec", t.log_every / dt, self.step)
                    if not all(math.isfinite(v) for v in msg.values()):
                        print("WARNING: non-finite loss detected", msg)
                    mem = torch.cuda.max_memory_allocated() / 2**30 if self.device.type == "cuda" else 0.0
                    self.writer.add_scalar("perf/max_mem_gb", mem, self.step)
                    print(f"step {self.step} ep {self.epoch} " + " ".join(f"{k}={v:.3f}" for k, v in msg.items())
                          + f" | {t.log_every / dt:.2f} it/s | mem {mem:.2f}GB", flush=True)
                    acc, t0 = {}, time.time()
                if self.step % t.eval_every == 0:
                    if self.is_main:
                        metrics = self.evaluate()
                        for k, v in metrics.items():
                            self.writer.add_scalar(f"val/{k}", v, self.step)
                        print(f"[eval] step {self.step} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
                        if metrics["val_mel_l1"] < self.best_val:
                            self.best_val = metrics["val_mel_l1"]
                            self.save("best.pt")
                    self.net_g.train()
                    if self.ddp:
                        dist.barrier()
                if self.step % t.save_every == 0:
                    self.save()
                if self.step >= t.max_steps or self._stop:
                    break
            self.epoch += 1
        self.save()
        if self.is_main:
            print(f"finished at step {self.step}; checkpoints in {self.ckpt_dir}")
            self.writer.close()
        if self.ddp:
            dist.destroy_process_group()
