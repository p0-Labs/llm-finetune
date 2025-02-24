## benchmarks
we literally just have to get better than that for an example, >= 30% would be ideal see:
https://refact.ai/blog/2023/introducing-refact-code-llm/

### llama-3.2-1b (ollama)
params:
```
'temperature': 0.7,
'top_p': 0.9,
'max_tokens': 400
```

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

### llama-3.2-1b-instruct (local weights)
params:
```
'temperature': 0.7,
'top_p': 0.9,
'max_tokens': 100
```

run 1:
```
total problems: 164
solved problems: 31
success rate: 18.90%
total time: 810.95 seconds
```
