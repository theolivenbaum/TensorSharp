# djev parity fixtures

| file | what | produced by |
|---|---|---|
| `djev_fixtures.json` | eight requests run through djev's own engine against a mock vLLM: every read it made and the response it built | `generate_djev_fixtures.py <djev-dev>` |
| `gemma_fixtures.json` | what djev compiles and sends for each of jevbench's 231 public decisions with DiffusionGemma's tokenizer | `generate_gemma_fixtures.py <djev-dev> <jevbench> <tokenizer-dir>` |
| `gemma_chat_template.jinja` | `chat_template.jinja` of `google/diffusiongemma-26B-A4B-it` at revision `f7f5b7f5fa82ffc52addd066915886d497f5517b` (the revision djev pins), unmodified | Google; Apache-2.0 per the model card |

djev is Apache-2.0 (<https://github.com/Davipar/djev-dev>); the generators import it from a checkout and do not
copy it. `gemma_fixtures.json` stores only token ids, hashes and, for eleven tasks, the prompt text and request
of jevbench's public MIT-licensed items. The tests that read these files are `DjevDecisionTests` and
`DjevGemmaParityTests`.
