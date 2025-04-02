import abc


class Model(abc.ABC):
    def get_repo(self, target_lang):
        ...

    def create(self, model_path):
        ...

    def translate_batch(self, texts, from_lang, to_lang, max_tok_alpha, params):
        ...


class GenericModel(Model):
    """
    Lower level than pipe to properly run batching
    See https://github.com/meta-llama/llama3/issues/114#issuecomment-2127131096
    """

    def __init__(self):
        self.padding_side = "left"
        self.dtype = 'bfloat16'

    def create(self, model_path):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # use bfloat16 for supported models and GPUS
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=self.dtype, device_map="auto"
        )
        # gemma returns empty strings with default padding side (left)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=self.padding_side)

    def get_chat_prompt(self, prompt):
        ...

    def translate_batch(self, texts, from_lang, to_lang, max_tok_alpha, params):
        import torch
        import toolz

        def get_prompt(text, from_lang, to_lang):
            prompt = (
                f"Translate this from {from_lang} to {to_lang}:\n{from_lang}: {text}\n{to_lang}:"
            )
            chat_style_prompt = self.get_chat_prompt(prompt)
            chat_prompt = self.tokenizer.apply_chat_template(
                chat_style_prompt, tokenize=False, add_generation_prompt=True
            )
            return chat_prompt

        prompts = [get_prompt(text, from_lang, to_lang) for text in texts]
        inputs = self.tokenizer(
            # pad to the longest sequence in a batch, never truncate (default)
            # padding negatively affects quality, ideally the input should be sorted by length and split to batches to make lines of similar size
            prompts,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        ).to(self.model.device)

        max_input_tokens = max(len(tokens) for tokens in self.tokenizer(texts).input_ids)
        max_new_tokens = int(max_tok_alpha * max_input_tokens)

        # Translation
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **params)
            outputs = self.parse_outputs(
                self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            )

        num_candidates = params.get("num_return_sequences", 1)
        assert len(outputs) % num_candidates == 0
        batch_results = (
            list(toolz.partition_all(num_candidates, outputs)) if num_candidates > 1 else outputs
        )
        return batch_results

    def parse_outputs(self, outputs):
        # it returns ...\n{output}
        processed_outputs = []
        for output in outputs:
            parts = output.strip().split("\n")
            processed_outputs.append(parts[-1].strip())
        return processed_outputs


class Gemma(GenericModel):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.padding_side = "right"

    def get_repo(self, target_lang):
        return f"google/gemma-3-{self.size}b-it"

    def get_chat_prompt(self, prompt):
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Respond with a translation only! Always reply with a translation, even when you are not sure.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]


class Llama3(GenericModel):
    def __init__(self, version, size):
        super().__init__()
        self.version = version
        self.size = size

    def get_repo(self, target_lang):
        return f"meta-llama/Llama-3.{self.version}-{self.size}B-Instruct"

    def create(self, model_path):
        super().create(model_path)
        # https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/discussions/39
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # no bfloat16 support
        # https://stackoverflow.com/questions/77803696/runtimeerror-cutlassf-no-kernel-found-to-launch-when-running-huggingface-tran
        # import torch
        # torch.backends.cuda.enable_mem_efficient_sdp(False)
        # torch.backends.cuda.enable_flash_sdp(False)

    def get_chat_prompt(self, prompt):
        return [
            {
                "role": "system",
                "content": "Respond with a translation only! Always reply with a translation, even when you are not sure.",
            },
            {"role": "user", "content": prompt},
        ]

class DeepSeek(Llama3):
    """
    DeepSeekR1 finetunes need a lot larger max_new_tokens because it's a reasoning model that generates extra thinking tokens
    """
    def __init__(self, version, size):
        super().__init__(version, size)

    def get_repo(self, target_lang):
        return f"deepseek-ai/DeepSeek-R1-Distill-{self.version}-{self.size}B"




class XAlma(GenericModel):
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

    def __init__(self):
        super().__init__()

        self.padding_side = "left"
        self.dtype = 'float16'

    def get_repo(self, target_lang):
        group_id = self.LANG2GROUP[target_lang]
        return f"haoranxu/X-ALMA-13B-Group{group_id}"

    def get_chat_prompt(self, prompt):
        return [{"role": "user", "content": prompt}]

    def parse_outputs(self, outputs):
        processed_outputs = []
        for output in outputs:
            parts = output.split("[/INST]")
            assert len(parts) == 2
            processed_outputs.append(parts[-1])
        return processed_outputs


class Runner:
    MODELS = {
        "llama-3-70b": Llama3(3, 70),
        "llama-3-8b": Llama3(1, 8),
        "x-alma-13b": XAlma(),
        "gemma-3-27b": Gemma(27),
        "gemma-3-12b": Gemma(12),
        "deepseek-llama-8b": DeepSeek('Llama', 8),
        "deepseek-llama-70b": DeepSeek('Llama', 70),
        "deepseek-qwen-14b": DeepSeek('Qwen', 14),
    }

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
            raise
