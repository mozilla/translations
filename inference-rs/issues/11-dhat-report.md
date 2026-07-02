# dhat report

https://github.com/nnethercote/dhat-rs

I want some dhat reports built to understand where memory is being allocated. We can save these out to inference-rs/artifacts/dhat* of some kind. Ultimately I use the Firefox Profiler to visualize these, but this is a human activity. From your side we can collect and report on things, and then let a human see the results. Let's do this as a `--memory-report` flag for inference-rs/scripts/translate.py that accepts no args, and a cfg feature to keep compiles small and targeted
