from datetime import datetime, timedelta

from pipeline.eval.translators import BergamotModel, BergamotPivotTranslator, BergamotTranslator


def test_pivot_prefers_release_models(monkeypatch):
    now = datetime.now()
    models = {
        ("de", "en"): [
            BergamotModel("de", "en", "released-de-en", now),
            BergamotModel("de", "en", "newer-de-en", now + timedelta(days=1)),
        ],
        ("en", "fr"): [
            BergamotModel("en", "fr", "released-en-fr", now),
            BergamotModel("en", "fr", "newer-en-fr", now + timedelta(days=1)),
        ],
    }
    hashes = {"released-de-en": "hash-de-en", "released-en-fr": "hash-en-fr"}

    def list_models(_bucket: str, src: str | None = None, trg: str | None = None):
        assert src is not None and trg is not None
        return models[(src, trg)]

    monkeypatch.setattr(
        BergamotTranslator,
        "list_all_models",
        staticmethod(list_models),
    )

    class Response:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    def get(url):
        if url == BergamotTranslator.release_models_url:
            return Response(
                {
                    "data": [
                        {
                            "sourceLanguage": "de",
                            "targetLanguage": "en",
                            "fileType": "model",
                            "filter_expression": "",
                            "decompressedHash": "hash-de-en",
                        },
                        {
                            "sourceLanguage": "en",
                            "targetLanguage": "fr",
                            "fileType": "model",
                            "filter_expression": "",
                            "decompressedHash": "hash-en-fr",
                        },
                        {
                            "sourceLanguage": "de",
                            "targetLanguage": "en",
                            "fileType": "model",
                            "filter_expression": "env.channel == 'nightly'",
                            "decompressedHash": "hash-newer-de-en",
                        },
                    ]
                }
            )
        name = url.split("/")[-3]
        return Response({"hash": hashes.get(name, f"hash-{name}")})

    monkeypatch.setattr("pipeline.eval.translators.requests.get", get)

    translator = BergamotPivotTranslator("de", "fr", "bucket", "bergamot-translator")

    assert translator.list_models() == ["released-de-en---released-en-fr"]


def test_pivot_falls_back_to_latest_model_without_release(monkeypatch):
    now = datetime.now()
    models = {
        ("de", "en"): [BergamotModel("de", "en", "latest-de-en", now)],
        ("en", "fr"): [BergamotModel("en", "fr", "latest-en-fr", now)],
    }

    def list_models(_bucket: str, src: str | None = None, trg: str | None = None):
        assert src is not None and trg is not None
        return models[(src, trg)]

    monkeypatch.setattr(
        BergamotTranslator,
        "list_all_models",
        staticmethod(list_models),
    )
    monkeypatch.setattr(BergamotTranslator, "list_release_models", lambda _self: [])

    translator = BergamotPivotTranslator("de", "fr", "bucket", "bergamot-translator")

    assert translator.list_models() == ["latest-de-en---latest-en-fr"]
