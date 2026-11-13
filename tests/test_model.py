import torch

from persian_tts.config import ModelConfig
from persian_tts.models import MultiPeriodDiscriminator, SynthesizerTrn
from persian_tts.text import SYMBOLS, text_to_ids

SMALL = ModelConfig(hidden_channels=32, filter_channels=64, inter_channels=32, n_layers=2,
                    upsample_initial_channel=64, gin_channels=16, dp_filter_channels=32)


def _model(n_spk=3):
    torch.manual_seed(0)
    return SynthesizerTrn(len(SYMBOLS), 513, 8, n_spk, SMALL)


def test_train_forward_and_disc():
    m = _model()
    x = torch.randint(2, len(SYMBOLS), (2, 20))
    xl = torch.tensor([20, 15])
    y = torch.randn(2, 513, 60).abs()
    yl = torch.tensor([60, 50])
    o, l_dur, attn, ids, *_ = m(x, xl, y, yl, torch.tensor([0, 2]))
    assert o.shape == (2, 1, 8 * 256)
    assert attn.sum(2)[0, 0].sum() == 60  # alignment covers every frame
    d = MultiPeriodDiscriminator()
    r, g, *_ = d(torch.randn(2, 1, 2048), o)
    assert len(r) == 6 and torch.isfinite(l_dur)


def test_infer_variable_lengths():
    m = _model().eval()
    m.remove_weight_norm()
    for text in ["سلام", "امروز هوا خیلی خوب است و ما به پارک می‌رویم."]:
        ids = torch.LongTensor([text_to_ids(text)])
        o, yl = m.infer(ids, torch.LongTensor([ids.shape[1]]), sid=torch.LongTensor([1]))
        assert o.shape[-1] == yl.item() * 256
