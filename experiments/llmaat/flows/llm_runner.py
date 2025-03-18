import abc

import toolz
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


class Model(abc.ABC):
    def create(self, target_lang):
        ...

    def transalte_batch(self, texts, from_lang, to_lang):
        ...


class XAlma(Model):
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

    def create(self, target_lang):
        group_id = self.LANG2GROUP[target_lang]
        self.model = AutoModelForCausalLM.from_pretrained(
            f"haoranxu/X-ALMA-13B-Group{group_id}", torch_dtype=torch.float16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"haoranxu/X-ALMA-13B-Group{group_id}", padding_side="left"
        )

    def transalte_batch(self, texts, from_lang, to_lang):
        def get_prompt(text, from_lang, to_lang):
            prompt = (
                f"Translate this from {from_lang} to {to_lang}:\n{from_lang}: {text}\n{to_lang}:"
            )
            chat_style_prompt = [{"role": "user", "content": prompt}]
            chat_prompt = self.tokenizer.apply_chat_template(
                chat_style_prompt, tokenize=False, add_generation_prompt=True
            )
            return chat_prompt

        prompts = [get_prompt(text, from_lang, to_lang) for text in texts]
        input_ids = self.tokenizer(
            prompts, return_tensors="pt", padding=True, max_length=600, truncation=True
        ).input_ids.cuda()

        # Translation
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                num_beams=5,
                max_new_tokens=600,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        results = []
        # print(outputs)
        for output in outputs:
            parts = output.split("[/INST]")
            assert len(parts) == 2
            results.append(parts[-1])
        return results


class Llama3(Model):
    def __init__(self, size):
        self.size = size

    def create(self, target_lang):
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        # https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/discussions/39
        self.pipeline.tokenizer.pad_token_id = self.pipeline.model.config.eos_token_id[0]
        # https://stackoverflow.com/questions/77803696/runtimeerror-cutlassf-no-kernel-found-to-launch-when-running-huggingface-tran
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)

    def transalte_batch(self, texts, from_lang, to_lang):
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
        outputs = self.pipeline(
            messages,
            max_new_tokens=200,
            temperature=0.6,
            top_p=0.9,
            num_beams=5,
            batch_size=len(texts),
        )
        return [output[0]["generated_text"][-1]["content"].strip() for output in outputs]


class Runner:
    MODELS = {"llama3-8b": Llama3(8), "llama3-70b": Llama3(70), "x-alma-13b": XAlma()}

    def __init__(self, model, target_lang):
        self.model = self.MODELS[model]
        print("creating model")
        self.model.create(target_lang)

    def translate(self, texts, from_lang, to_lang, batch_size=32):
        try:
            batch_res = []
            for batch in tqdm(list(toolz.partition_all(batch_size, texts))):
                batch_res.append(self.model.translate_batch(batch, from_lang, to_lang))

            translations = []
            for res in batch_res:
                translations.extend(res)

            return translations
        except Exception as ex:
            print(f"Error while translating: {ex}")
