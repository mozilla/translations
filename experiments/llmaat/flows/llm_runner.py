import abc


class Model(abc.ABC):
    def get_repo(self, target_lang):
        ...

    def create(self, model_path):
        ...

    def translate_batch(self, texts, from_lang, to_lang, max_tok_alpha, params):
        ...


class GenericModel(Model):
    def create(self, model_path):
        from transformers import pipeline
        import torch

        self.pipe = pipeline(
            "text-generation", model=model_path, device="cuda", torch_dtype=torch.bfloat16
        )

    def translate_batch(self, texts, from_lang, to_lang, max_tok_alpha, params):
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

        outputs = self.pipe(messages, max_new_tokens=600, **params)
        return [output[0]["generated_text"][-1]["content"].strip() for output in outputs]


class Gemma(GenericModel):
    def get_repo(self, target_lang):
        return "google/gemma-3-1b-it"


class Llama3(Model):
    def get_repo(self, target_lang):
        return "meta-llama/Llama-3.3-70B-Instruct"

    def create(self, model_path):
        from transformers import pipeline
        import torch

        self.pipeline = pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        # https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/discussions/39
        self.pipeline.tokenizer.pad_token_id = self.pipeline.model.config.eos_token_id[0]
        # https://stackoverflow.com/questions/77803696/runtimeerror-cutlassf-no-kernel-found-to-launch-when-running-huggingface-tran
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)

    def translate_batch(self, texts, from_lang, to_lang, max_tok_alpha, params):
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
        outputs = self.pipeline(messages, batch_size=len(texts), **params)
        return [output[0]["generated_text"][-1]["content"].strip() for output in outputs]


class XAlma(Model):
    """
    https://huggingface.co/haoranxu/X-ALMA

    Params from the example:
    num_beams=5, do_sample=True, temperature=0.6, top_p=0.9

    From HF:
        greedy decoding if num_beams=1 and do_sample=False
        multinomial sampling if num_beams=1 and do_sample=True
        beam-search decoding if num_beams>1 and do_sample=False
        beam-search multinomial sampling if num_beams>1 and do_sample=True
    """

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

    def get_repo(self, target_lang):
        group_id = self.LANG2GROUP[target_lang]
        return f"haoranxu/X-ALMA-13B-Group{group_id}"

    def create(self, model_path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

    def translate_batch(self, texts, from_lang, to_lang, max_tok_alpha, params):
        import torch

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
            prompts, return_tensors="pt", padding=True, max_length=512, truncation=False
        ).input_ids.cuda()

        max_input_tokens = max(len(tokens) for tokens in self.tokenizer(texts).input_ids)
        max_new_tokens = int(max_tok_alpha * max_input_tokens)

        # Translation
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens, **params
            )
            outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        batch_results = []
        for output in outputs:
            if isinstance(output, str):
                parts = output.split("[/INST]")
                assert len(parts) == 2
                batch_results.append(parts[-1])
            else:
                # num_return_sequences > 1
                cand_results = []
                for candidate in output:
                    parts = candidate.split("[/INST]")
                    assert len(parts) == 2
                    cand_results.append(parts[-1])
                batch_results.append(cand_results)

        return batch_results


class Runner:
    MODELS = {"llama3-70b": Llama3(), "x-alma-13b": XAlma(), "gemma-3-1b": Gemma()}

    def __init__(self, model_name):
        self.model = self.MODELS[model_name]

    def get_repo(self, target_lang):
        return self.model.get_repo(target_lang)

    def create(self, model_path):
        self.model.create(model_path)

    def translate(self, texts, from_lang, to_lang, batch_size, max_tok_alpha, params):
        import toolz
        from tqdm import tqdm

        try:
            batch_res = []
            for batch in tqdm(list(toolz.partition_all(batch_size, texts))):
                batch_res.append(
                    self.model.translate_batch(batch, from_lang, to_lang, max_tok_alpha, params)
                )

            translations = []
            for res in batch_res:
                translations.extend(res)

            return translations
        except Exception as ex:
            print(f"Error while translating: {ex}")
