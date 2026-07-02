# Rust-only `fxtranslate` Package

Probably sequenced at the end of this work. I want a fully-powered `fxtranslate` package built here that has batteries included. Basically if you cargo install it, I want to be able to do the following:

Enumerate the the list of models available in Remote Settings. Let's not worry about langauge display names as that sounds like a bigger dependency.

Download and manage the model files in a local cache of some kind.

Run translations by the stdout piping per-sentence inputs similar to how Marian already does it.

A fully interactive translator that works when you type in it. Enter sends in a translation, and then outputs the translation in your terminal. This is a minimal pretty and ergonomic CLI tool. I want this to be dependency free as possible, just as a nice "kick the tires" kind of feature.

DO NOT PUBLISH ANYTHING, but come up with a publish plan as a durable markdown artifact afterwards.
