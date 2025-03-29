import abc


class Model(abc.ABC):
    def get_repo(self, target_lang):
        ...

    def create(self, model_path):
        ...

    def translate_batch(self, texts, from_lang, to_lang, max_tok_alpha, params):
        ...


class PipelineModel(Model):
    def create(self, model_path):
        from transformers import pipeline
        import torch

        self.pipe = pipeline(
            "text-generation",
            model=model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

    def translate_batch(self, texts, from_lang, to_lang, max_tok_alpha, params):
        messages = [[
            {
                "role": "system",
                "content": [{"type": "text",
                             "text": "Respond with a translation only! Always reply with a translation, even when you are not sure."}]
            },
            {
                "role": "user",
                "content": [{"type": "text",
                             "text": f"Translate this from {from_lang} to {to_lang}:\n{from_lang}: {text}\n{to_lang}:"}]
            }
        ] for text in texts]

        tokenizer_args = {"truncation": False, "padding": 'longest'}
        params.update(tokenizer_args)
        max_input_tokens = max(len(tokens) for tokens in self.pipe.tokenizer(texts).input_ids)
        max_new_tokens = int(max_tok_alpha * max_input_tokens)

        outputs = self.pipe(messages, batch_size=len(messages), max_new_tokens=max_new_tokens, **params)

        if params.get("num_return_sequences", 1) > 1:
            new_outputs = []
            for output in outputs:
                cands = []
                for cand in output:
                    cands.append(cand["generated_text"][-1]["content"].strip())
                new_outputs.append(cands)

        else:
            new_outputs = [output[0]["generated_text"][-1]["content"].strip() for output in outputs]
        return new_outputs

class GenericModel(Model):
    def create(self, model_path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # use bfloat16 for supported models and GPUS
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)


    def translate_batch(self, texts, from_lang, to_lang, max_tok_alpha, params):
        import torch
        import toolz

        def get_prompt(text, from_lang, to_lang):
            prompt = (
                f"Translate this from {from_lang} to {to_lang}:\n{from_lang}: {text}\n{to_lang}:"
            )
            chat_style_prompt = [{
                "role": "system",
                "content": "Respond with a translation only! Always reply with a translation, even when you are not sure.",
            },
                {"role": "user", "content": prompt}]
            chat_prompt = self.tokenizer.apply_chat_template(
                chat_style_prompt, tokenize=False, add_generation_prompt=True
            )
            return chat_prompt

        prompts = [get_prompt(text, from_lang, to_lang) for text in texts]
        input_ids = self.tokenizer(
            # pad to the longest sequence in a batch, never truncate (default)
            prompts,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        ).input_ids.to(self.model.device)

        max_input_tokens = max(len(tokens) for tokens in self.tokenizer(texts).input_ids)
        max_new_tokens = int(max_tok_alpha * max_input_tokens)

        # Translation
        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens, **params
            )
            outputs = self.parse_outputs(self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

        num_candidates = params.get("num_return_sequences", 1)
        assert len(outputs) % num_candidates == 0
        batch_results = (
            list(toolz.partition_all(num_candidates, outputs))
            if num_candidates > 1
            else outputs
        )
        return batch_results


    def parse_outputs(self, outputs: list) -> list:
        ...


class GemmaPipeline(PipelineModel):
    def __init__(self, size):
        self.size = size

    def get_repo(self, target_lang):
        return f"google/gemma-3-{self.size}b-it"

class Gemma(GenericModel):
    def __init__(self, size):
        self.size = size

    def get_repo(self, target_lang):
        return f"google/gemma-3-{self.size}b-it"

    def parse_outputs(self, outputs):
        # it always returns ...\nmodel\n{output}
        processed_outputs = []
        for output in outputs:
            parts = output.strip().split('\nmodel\n')[-1]
            assert len(parts) == 2
            processed_outputs.append(parts[1])

class Llama3(PipelineModel):
    def get_repo(self, target_lang):
        return "meta-llama/Llama-3.3-70B-Instruct"

    def create(self, model_path):
        import torch

        super(GenericModel).create(model_path)
        # https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/discussions/39
        self.pipe.tokenizer.pad_token_id = self.pipe.model.config.eos_token_id[0]
        # https://stackoverflow.com/questions/77803696/runtimeerror-cutlassf-no-kernel-found-to-launch-when-running-huggingface-tran
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)


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
        import toolz

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
            # pad to the longest sequence in a batch, never truncate (default)
            prompts,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        ).input_ids.cuda()

        max_input_tokens = max(len(tokens) for tokens in self.tokenizer(texts).input_ids)
        max_new_tokens = int(max_tok_alpha * max_input_tokens)

        # Translation
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens, **params
            )
            outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        processed_outputs = []
        for output in outputs:
            parts = output.split("[/INST]")
            assert len(parts) == 2
            processed_outputs.append(parts[-1])

        num_candidates = params.get("num_return_sequences", 1)
        assert len(processed_outputs) % num_candidates == 0
        batch_results = (
            list(toolz.partition_all(num_candidates, processed_outputs))
            if num_candidates > 1
            else processed_outputs
        )
        return batch_results


class Runner:
    MODELS = {"llama3-70b": Llama3(), "x-alma-13b": XAlma(), "gemma-3-27b": GemmaPipeline(27), "gemma-3-12b": GemmaPipeline(12)}

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
