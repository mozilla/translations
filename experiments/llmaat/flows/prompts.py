
def chat_gpt(text, from_lang, to_lang):
    return [
        {
            "role": "system",
            "content": "You are a professional translator. Translate the user's text from "
                       f"{from_lang} to {to_lang} and output only the translated text."
        },
        {
            "role": "user",
            "content": f"{from_lang}: {text}\n\n{to_lang}:"
        }
    ]



def basic(text, from_lang, to_lang):
    prompt = f"Translate this from {from_lang} to {to_lang}:\n{from_lang}: {text}\n{to_lang}:"
    return [
        {
            "role": "system",
            "content": "Respond with a translation only! Always reply with a translation, even when you are not sure.",
        },
        {"role": "user", "content": prompt},
    ]


def wmt24pp(text, from_lang, to_lang):
    prompt = f"Translate this from {from_lang} to {to_lang}:\n{from_lang}: {text}\n{to_lang}:"
    return [
        {
            "role": "system",
            "content": "Respond with a translation only! Always reply with a translation, even when you are not sure.",
        },
        {"role": "user", "content": prompt},
    ]
