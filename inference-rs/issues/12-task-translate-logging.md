# Logging in the translate task.


```
➤ task inference-rs:translate -- en es --text "Hello world! This is a translation of some text."
task: [inference-rs:translate] python3 /Users/greg/dev/translations/inference-rs/scripts/translate.py en es --text 'Hello world! This is a translation of some text.'
[run] cargo run --quiet --manifest-path /Users/greg/dev/translations/inference-rs/Cargo.toml -- translate data/models/enes/model.enes.intgemm.alphas.bin data/models/enes/vocab.enes.spm data/models/enes/vocab.enes.spm
Hola mundo! Esta es una traducción de algún texto.
```

This task takes a long time to run with long time to first token. I want better logging here to understand and report on timing. I'm assuming there are some long oeprations we can report on with timing.

Do some design work and specify here in this issue what the output design should look like before diving into the implementation.
