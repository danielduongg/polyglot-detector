"""Tests for the in-browser language detector model."""
import json
import train  # train.py has a __main__ guard, safe to import

def _model(): return json.load(open("model/lang_model.json"))

def test_model_shape():
    m=_model()
    assert set(m["langs"])=={"en","fr","vi","es"}
    assert m["V"]>0 and m["nmin"]>=1 and m["nmax"]>=m["nmin"]

def test_classifies_each_language():
    m=_model()
    cases={
        "the quick brown fox jumps over the lazy dog":"en",
        "le chat dort sur le canapé pendant la journée":"fr",
        "tôi rất thích ăn phở vào buổi sáng hôm nay":"vi",
        "el perro corre muy rápido por el parque grande":"es",
    }
    for text,lang in cases.items():
        assert train.predict(m,text)==lang, (text, train.predict(m,text))

def test_reported_accuracy_high():
    import json as _j
    acc=_j.load(open("results/metrics.json"))["held_out_accuracy"]
    assert acc>0.9
