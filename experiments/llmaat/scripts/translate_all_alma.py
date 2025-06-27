import os

import toolz
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

GROUP2LANG = {
    1: ["da", "nl", "de", "is", "no", "sv", "af"],
    2: ["ca", "ro", "gl", "it", "pt", "es"],
    3: ["bg", "mk", "sr", "uk", "ru"],
    4: ["id", "ms", "th", "vi", "mg", "fr"],
    5: ["hu", "el", "cs", "pl", "lt", "lv"],
    6: ["ka", "zh", "ja", "ko", "fi", "et"],
    7: ["gu", "hi", "mr", "ne", "ur"],
    8: ["az", "kk", "ky", "tr", "uz", "ar", "he", "fa"],
}
LANG2GROUP = {lang: str(group) for group, langs in GROUP2LANG.items() for lang in langs}
group_id = LANG2GROUP["ru"]

model = AutoModelForCausalLM.from_pretrained(
    f"haoranxu/X-ALMA-13B-Group{group_id}", torch_dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    f"haoranxu/X-ALMA-13B-Group{group_id}", padding_side="left"
)


def get_prompt(text, from_lang, to_lang):
    prompt = f"Translate this from {from_lang} to {to_lang}:\n{from_lang}: {text}\n{to_lang}:"
    chat_style_prompt = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        chat_style_prompt, tokenize=False, add_generation_prompt=True
    )
    return chat_prompt


def translate_batch(texts, from_lang, to_lang):
    try:
        prompts = [get_prompt(text, from_lang, to_lang) for text in texts]
        input_ids = tokenizer(
            prompts, return_tensors="pt", padding=True, max_length=600, truncation=True
        ).input_ids.cuda()

        # Translation
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                num_beams=5,
                max_new_tokens=600,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        results = []
        # print(outputs)
        for output in outputs:
            parts = output.split("[/INST]")
            assert len(parts) == 2
            results.append(parts[-1])
        return results
    except Exception as ex:
        print(ex)
        print(output)
        raise


langs = [
    ("ru", "en", "Russian", "English"),
    ("en", "ru", "English", "Russian"),
    ("en", "de", "English", "German"),
    ("de", "en", "German", "English"),
]

datasets = ["wmt22", "wmt23"]
model_name = "x-alma"

BATCH_SIZE = 10

for dataset in datasets:
    for from_code, to_code, from_lang, to_lang in langs:
        output = f"{model_name}/{from_code}-{to_code}/{dataset}.{from_code}-{to_code}.translations.{to_code}"
        if os.path.isfile(output):
            print(f"Skipping, {output} already exists")
            continue

        print(f"translating {from_lang} to {to_lang} for {dataset}")

        with open(
            f"{model_name}/{from_code}-{to_code}/{dataset}.{from_code}-{to_code}.{from_code}"
        ) as f:
            lines = [l.strip() for l in f.readlines()]

        try:
            batch_res = []
            for batch in tqdm(list(toolz.partition_all(BATCH_SIZE, lines))):
                batch_res.append(translate_batch(batch, from_lang, to_lang))

            translations = []
            for res in batch_res:
                translations.extend(res)

            with open(output, "w") as f:
                f.write("\n".join(translations))
        except Exception as ex:
            print(f"Error while translating: {ex}")
