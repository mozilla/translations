# Sentence Piece Oracle

We need really good sentence piece verification. Let's design a durable oracle that can iterate through edge cases with tokenization. It should load in some kind of substantial real-world corpus, maybe by sampling and committing from NLLB.

I would assume a diverse set of 10,000 examples would give me some confidence. There's some things like unicode byte fallbacking that would be good test.

I believe we're missing normalization behavior as well.

> Tokenizer normalization gaps → will bite on accented/CJK, invisible on my Latin sample.
