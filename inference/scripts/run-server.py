"""
fast-api server using inference/build/marian-decoder and downloaded models from https://mozilla.github.io/translations/firefox-models/

based on code found on https://agentbus.sh/posts/how-to-build-a-language-detection-and-translation-pipeline/

Manually download models to ./data/models/{src}-{tgt}
    - lex.50.50.{src}{tgt}.s2t.bin # unused in this script
    - model.{src}{tgt}.intgemm.alphas.bin
    - vocab.{src}{tgt}.spm # if single file
        - srcvocab.{src}{tgt}.spm # if two files
        - trgvocab.{src}{tgt}.spm # if two files
"""

from operator import attrgetter
import subprocess
import pathlib
from contextlib import asynccontextmanager
from lingua import Language, LanguageDetectorBuilder
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

LANG_TO_ISO = {lang: lang.iso_code_639_1.name.lower() for lang in Language.all()}
detector = LanguageDetectorBuilder.from_all_languages().build()

def load_model(src: str, tgt: str):
    dirname = f"./data/models_inference/{src}-{tgt}"
    if src in ["zh-Hans", "zh-Hant"]:
        src = "zh"
    if tgt in ["zh-Hans", "zh-Hant"]:
        tgt = "zh"
    modelname = f"{dirname}/model.{src}{tgt}.intgemm.alphas.bin"
    vocfiles = list(pathlib.Path(dirname).glob(f'*vocab.{src}{tgt}.spm'))
    if len(vocfiles) == 1 and vocfiles[0].name == f'vocab.{src}{tgt}.spm':
        src_voc = f'{dirname}/vocab.{src}{tgt}.spm'
        trg_voc = f'{dirname}/vocab.{src}{tgt}.spm'
    else:
        src_voc = f'{dirname}/srcvocab.{src}{tgt}.spm'
        trg_voc = f'{dirname}/trgvocab.{src}{tgt}.spm'
    loads = (src_voc, trg_voc, modelname)
    return loads

def do_translate(text: str, src: str, tgt: str) -> str:
    try:
        src_voc, trg_voc, modelname = load_model(src, tgt)
    except Exception:
        # Pivot through English
        srcvocab_en, trgvocab_en, mod_en = load_model(src, "en")
        command = (
            f"echo {text} | "
            "./inference/build/marian-decoder "
            # f"--config {decoder} "
            f"--models {mod_en} "
            f"--vocabs {srcvocab_en} {trgvocab_en} "
            "--quiet"
        )
        en_text = subprocess.check_output(command,stderr=subprocess.STDOUT,shell=True)
        if tgt == "en":
            return en_text
        srcvocab_tgt, trgvocab_tgt, mod_tgt = load_model("en", tgt)
        command = (
            f'echo "{en_text}" | '
            "./inference/build/marian-decoder "
            # f"--config {decoder} "
            f"--models {mod_tgt} "
            f"--vocabs {srcvocab_tgt} {trgvocab_tgt} "
            "--quiet"
        )
        tgt_text = subprocess.check_output(command,stderr=subprocess.STDOUT,shell=True)
        return tgt_text
    command = (
        f'echo "{text}" | '
        "./inference/build/marian-decoder "
        # f"--config {decoder} "
        f"--models {modelname} "
        f"--vocabs {src_voc} {trg_voc} "
        "--quiet"
    )
    tgt_text = subprocess.check_output(command,stderr=subprocess.STDOUT,shell=True)
    return tgt_text

class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    target_lang: str = Field(default="en")
    source_lang: str = Field(default="ja")

class TranslateResponse(BaseModel):
    detected_language: str
    confidence: float
    source_lang: str
    target_lang: str
    translated_text: str

@app.post("/translate", response_model=TranslateResponse)
def translate_endpoint(req: TranslateRequest):
    src_lang = req.source_lang
    tgt_lang = req.target_lang
    detected_language = "N/A"
    confidence = 0.0
    if src_lang == "auto":
        confidences = detector.compute_language_confidence_values(req.text)
        if not confidences or all(conf.value < 0.25 for conf in confidences):
            raise HTTPException(400, "Could not detect language with sufficient confidence")

        detected_lang = max(confidences, key=attrgetter('value'))
        src_lang = LANG_TO_ISO.get(detected_lang.language)
        if src_lang is None:
            raise HTTPException(400, f"Unsupported language: {detected_lang.language}")
        detected_language = str(detected_lang.language)
        confidence = round(detected_lang.value, 4)
    if src_lang == "zh-CN" or src_lang == "zh":
        src_lang = "zh-Hans"
    if tgt_lang == "zh-CN" or tgt_lang == "zh":
        tgt_lang = "zh-Hans"
    if src_lang == "zh-TW":
        src_lang = "zh-Hant"
    if tgt_lang == "zh-TW":
        tgt_lang = "zh-Hant"
    if src_lang == req.target_lang:
        return TranslateResponse(
            detected_language=detected_language,
            confidence=confidence,
            source_lang=src_lang,
            target_lang=tgt_lang,
            translated_text=req.text,
        )
    translated = do_translate(req.text, src_lang, tgt_lang)
    return TranslateResponse(
        detected_language=detected_language,
        confidence=confidence,
        source_lang=src_lang,
        target_lang=tgt_lang,
        translated_text=translated,
    )

# poetry run uvicorn inference.scripts.run-server:app --host 0.0.0.0 --port 8080