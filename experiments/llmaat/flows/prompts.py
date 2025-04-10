from langs import LANGS


def chat_gpt(text, from_lang_code, to_lang_code):
    from_lang = LANGS[from_lang_code]["lang"]
    to_lang = LANGS[to_lang_code]["lang"]
    system_prompt = (
        "You are a professional translator. Translate the user's text from "
        f"{from_lang} to {to_lang} and output only the translated text."
    )
    user_prompt = f"{from_lang}: {text}\n\n{to_lang}:"
    return system_prompt, user_prompt


def basic(text, from_lang_code, to_lang_code):
    from_lang = LANGS[from_lang_code]["lang"]
    to_lang = LANGS[to_lang_code]["lang"]
    system_prompt = "Respond with a translation only! Always reply with a translation, even when you are not sure."
    user_prompt = f"Translate this from {from_lang} to {to_lang}:\n{from_lang}: {text}\n{to_lang}:"
    return system_prompt, user_prompt


def wmt24pp(text, from_lang_code, to_lang_code):
    region = LANGS[to_lang_code]["country"]
    from_lang = LANGS[from_lang_code]["lang"]
    to_lang = LANGS[to_lang_code]["lang"]
    system_prompt = (
        f"You are a professional {from_lang} to {to_lang} translator, tasked with providing translations suitable for "
        f"use in {region}. Your goal is to accurately convey the meaning and nuances of the original {from_lang} text "
        f"while adhering to {to_lang} grammar, vocabulary and cultural sensitivities."
    )
    user_prompt = (
        f"Please translate the following {from_lang} text into {to_lang} ({to_lang_code}):\n\n{text}\n\n"
        f"Produce only the {to_lang} translation, without any additional explanation or commentary."
    )
    return system_prompt, user_prompt


prompts = {"chat-gpt": chat_gpt, "basic": basic, "wmt24pp": wmt24pp}
