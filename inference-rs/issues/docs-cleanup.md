# Docs Cleanup

For high quality code generation, we need human-focused documentation. Your generative loop for building out the original files left comment artifacts strewn across.

## What to remove:

Tautological comments - Comments that repeat the implementation details and don't really add value. Comments should explain why things happen, not just repeat details. I didn't see a ton of these in the code comments, but it's something to keep an eye out for. I'm less concerned here for bigger comment rewriting but have it loaded in your context that this is important.

Self talk, and process comments - We don't need durable comments about _how_ this project was built. There is a lot of referencing of code gen process. Anytime it references a 01-*.md markdown file, this should be a red flag for something to clean up. It's fine to reference future clean-ups or design directions with TODO and a reference link. It's also OK to reference oracle implementations.

## Follow-up issue:

This should be sequenced at the top of the agentic workflow, but I want a new issue for you to file that handles this as a final validation step of an agentic flow.

## Planning

In planning let's put in some examples here for me to review.
