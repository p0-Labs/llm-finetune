## benchmarks
we literally just have to get better than that for an example, >= 30% would be ideal see:
https://refact.ai/blog/2023/introducing-refact-code-llm/

ph1 (microsoft) -> 1.3B -> 50.6% pass@1 on HumanEval

params for all runs:
```
'temperature': 0.7,
'top_p': 0.9,
'max_tokens': 400
```

### llama-3.2-1b (ollama)
run 1:
```
total problems: 164
solved problems: 44
success rate: 26.83%
total time: 507.79 seconds
```

run 2:
```
total problems: 164
solved problems: 42
success rate: 25.61%
total time: 509.35 seconds
```

avg: 26.22% (43 solved problems)

### llama-3.2-1b-instruct (local weights fp16)
run 1:
```
total problems: 164
solved problems: 44
success rate: 26.83%
total time: 778.09 seconds
```

### llama-3.2-1b-instruct-ft-python (local weights)
run 1:
```
total problems: 164
solved problems: 35
success rate: 21.34%
total time: 466.49 seconds
```

run 2: (with 600 max tokens)
```
total problems: 164
solved problems: 36
success rate: 21.95%
total time: 411.89 seconds
```
