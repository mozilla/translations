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


examples = {
    "en_US-ru_RU": [
        ("- Online: http://wrathofman.movie", "- Онлайн: http://wrathofman.movie"),
        ("Growth Opportunity Analysis - FWD", "Анализ возможностей роста - FWD"),
        (
            "Oh wow - those colors are gorgeous! Love the butterfly!ReplyDelete",
            "О, вау — какие потрясающие цвета! Обожаю бабочку!ОтветитьУдалить",
        ),
        (
            "When you need a break, order a festive cocktail and pizza from the menu, and watch your friends defeat their opponents.",
            "Когда вам понадобится перерыв, закажите праздничный коктейль и пиццу из меню и наблюдайте, как ваши друзья побеждают своих соперников.",
        ),
        (
            "In other trading, benchmark U.S. crude rose 34 cents to $52.61 a barrel in electronic trading on the New York Mercantile Exchange. It lost 86 cents to $52.27 per barrel on Friday. Brent crude, the international standard edged up 30 cents to $55.71 a barrel.",
            "В других торгах эталонная американская нефть подорожала на 34 цента до $52,61 за баррель в ходе электронных торгов на Нью-Йоркской товарной бирже. В пятницу она подешевела на 86 центов до $52,27 за баррель. Нефть Brent, международный стандарт, прибавила 30 центов до $55,71 за баррель.",
        ),
    ]
}


def no_omit_few_shot(text, from_lang_code, to_lang_code):
    from_lang = LANGS[from_lang_code]["lang"]
    to_lang = LANGS[to_lang_code]["lang"]
    ex_str = "\n\n".join(
        [
            f"{from_lang}: {src}\n{to_lang}: {trg}"
            for src, trg in examples[f"{from_lang_code}-{to_lang_code}"]
        ]
    )
    system_prompt = (
        f"""You are a professional translator. 
Translate the user's text from {from_lang} to {to_lang}. 
Respond with a translation only. 
Do not omit anything in the translation.
Preserve the meaning as close as possible to the original text, even if doesn't make sense. 
The generated translations will be used to train a neural machine translation model.

Examples:

"""
        + ex_str
    )
    user_prompt = f"{from_lang}: {text}\n{to_lang}: "
    return system_prompt, user_prompt


prompts = {
    "chat-gpt": chat_gpt,
    "basic": basic,
    "wmt24pp": wmt24pp,
    "noomit_fewshot": no_omit_few_shot,
}
