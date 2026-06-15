from src.recognition.evaluator import RecogEvaluator


def test_perfect_predictions_zero_cer_wer():
    ev = RecogEvaluator()
    ev.add(["hello world", "TESCO"], ["hello world", "TESCO"])
    m = ev.compute()
    assert m["cer"] == 0.0
    assert m["wer"] == 0.0
    assert m["n"] == 2


def test_single_char_diff_cer():
    ev = RecogEvaluator()
    ev.add(["hella"], ["hello"])
    m = ev.compute()
    # 1 edit / 5 chars = 0.2
    assert abs(m["cer"] - 0.2) < 1e-6


def test_worst_n_returns_highest_cer_first():
    ev = RecogEvaluator()
    ev.add(
        preds=["hello", "wxyz", "tesco"],
        gts=  ["hello", "abcd", "tescq"],
    )
    worst = ev.worst_n(2)
    assert len(worst) == 2
    # "wxyz" vs "abcd" => CER 1.0 (all 4 differ); "tesco" vs "tescq" => 0.2
    assert worst[0]["pred"] == "wxyz"
    assert worst[0]["gt"] == "abcd"
    assert worst[1]["pred"] == "tesco"


def test_empty_gt_does_not_divide_by_zero():
    ev = RecogEvaluator()
    ev.add([""], [""])
    m = ev.compute()
    assert m["cer"] == 0.0
    assert m["wer"] == 0.0
