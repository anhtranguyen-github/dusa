import torch

from src.recognition.crnn_tiny import CRNNTiny, ctc_greedy_decode, DEFAULT_CHARSET


def test_crnn_forward_shape():
    m = CRNNTiny(num_classes=len(DEFAULT_CHARSET) + 1)
    x = torch.zeros(2, 1, 32, 128)
    out = m(x)
    assert out.shape[1] == 2
    assert out.shape[2] == len(DEFAULT_CHARSET) + 1


def test_ctc_greedy_decode_collapses_repeats_and_drops_blank():
    blank = len(DEFAULT_CHARSET)
    a_idx = DEFAULT_CHARSET.index("a")
    b_idx = DEFAULT_CHARSET.index("b")
    fake_logits = torch.full((6, 1, len(DEFAULT_CHARSET) + 1), -1e3)
    seq = [a_idx, a_idx, blank, a_idx, b_idx, b_idx]
    for t, idx in enumerate(seq):
        fake_logits[t, 0, idx] = 0.0
    out = ctc_greedy_decode(fake_logits, DEFAULT_CHARSET)
    assert out == ["aab"]
