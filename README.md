## llm-finetune
testing/experimenting in google colab for now...

### Llama-3.2-1B model architecture
```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 2048)
    (layers): ModuleList(
      (0-15): 16 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (k_proj): Linear(in_features=2048, out_features=512, bias=False)
          (v_proj): Linear(in_features=2048, out_features=512, bias=False)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
          (up_proj): Linear(in_features=2048, out_features=8192, bias=False)
          (down_proj): Linear(in_features=8192, out_features=2048, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((2048,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=128256, bias=False)
)
```

### Llama-3.2-1B (ollama) HumanEval
single run humaneval score on mba m3 16gb:

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

we literally just have to get better than that for an example, >= 30% would be ideal see:
https://refact.ai/blog/2023/introducing-refact-code-llm/

reached 25.7% on BFCL V2 per some source online

### Benchmarking
- needs to be like this repo: `https://github.com/openai/human-eval`
- then we can just replace the human-eval python version for zig version
