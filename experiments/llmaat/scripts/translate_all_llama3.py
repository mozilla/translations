import os

os.environ["HUGGING_FACE_HUB_TOKEN"] = ""
import transformers
import torch
from tqdm import tqdm
import toolz

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)


# https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/discussions/39
pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id[0]
# https://stackoverflow.com/questions/77803696/runtimeerror-cutlassf-no-kernel-found-to-launch-when-running-huggingface-tran
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


def translate_batch(texts, from_lang, to_lang):
    prompts = [
        "Translate this from "
        + from_lang
        + " to "
        + to_lang
        + ":\n"
        + from_lang
        + ":\n "
        + text
        + "\n"
        + to_lang
        + ":\n"
        for text in texts
    ]
    messages = [
        [
            {
                "role": "system",
                "content": "Respond with the translation only! Always reply with a translation, even when you are not sure.",
            },
            {"role": "user", "content": prompt},
        ]
        for prompt in prompts
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=200,
        temperature=0.6,
        top_p=0.9,
        num_beams=5,
        batch_size=len(texts),
    )
    return [output[0]["generated_text"][-1]["content"].strip() for output in outputs]


langs = [
    ("ru", "en", "Russian", "English"),
    ("en", "ru", "English", "Russian"),
    ("en", "de", "English", "German"),
    ("de", "en", "German", "English"),
]

datasets = ["wmt22", "wmt23"]


BATCH_SIZE = 8

for dataset in datasets:
    for from_code, to_code, from_lang, to_lang in langs:
        output = (
            f"llama3/{from_code}-{to_code}/{dataset}.{from_code}-{to_code}.translations.{to_code}"
        )
        if os.path.isfile(output):
            print(f"Skipping, {output} already exists")
            continue

        print(f"translating {from_lang} to {to_lang} for {dataset}")

        with open(
            f"llama3/{from_code}-{to_code}/{dataset}.{from_code}-{to_code}.{from_code}"
        ) as f:
            lines = [l.strip() for l in f.readlines()]

        try:
            batch_res = []
            with tqdm(total=len(lines)) as pbar:
                for batch in list(toolz.partition_all(BATCH_SIZE, lines)):
                    batch_res.append(translate_batch(batch, from_lang, to_lang))
                    pbar.update(len(batch))

            translations = []
            for res in batch_res:
                translations.extend(res)

            with open(output, "w") as f:
                f.write("\n".join(translations))
        except Exception as ex:
            print(f"Error while translating: {ex}")
