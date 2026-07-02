# Split Vocab Oracle

We have done shared vocab only. We need to build and validate off of a shared vocab oracle. I believe CJK models have split vocabs. We need to use the remote settings model loading machinery or remote settings API to find a split vocab, ideally a CJK one, and then build an oracle we can validate from.

The outcomes here are a durable set of tests that can validate split vocab, and some kind of cheat-proof validation against the known good implementation.
