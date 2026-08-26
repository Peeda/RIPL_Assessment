# Evidence from failed generations

Kept because "the request looked fine and the model returned nothing usable" is
a claim that needs the actual response behind it, and because these cost money.

| file | what it shows |
|---|---|
| `gap_stub_24k_thinking.json` | `tool_choice: auto` restored the reasoning — `thinking_tokens` 23,947 over 350 s — and the model *still* returned `'placeholder'` in `reward_py`, `rationale` and `uncertainties` while writing a real 5,019-character `sampler_py`. This is the response that motivated the "One call" prompt section and the cross-attempt merge in `generate.py`. |

The four earlier generations, from before `tool_choice` was un-forced, are at
`~/ripl/t3/badresults*` on the laptop and are referenced from CLAUDE.md's
"A FORCED `tool_choice` ZEROES THE THINKING" section. They are not in git.
