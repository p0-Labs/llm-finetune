## notes

### LoRA
- [https://medium.com/@tayyibgondal2003/loralow-rank-adaptation-of-large-language-models-33f9d9d48984](informative article on it)
- the rank (r) determins the size of the adapters. say you have weight matrix W that's (10,000x20,000) and you choose
r = 8 then you split W into A and B that are (10,000x8) and (8x20,000) respectively such that if you multiply them, you
get the original W size again.
- as you increase rank, lora converges towards normal fine-tuning. higher rank uses more memory, but can retain more
information. lower rank less memory, but less information.
- usually r = 4-16 is good, but if the fine-tuning data differs significantly from the pre-training data, then you might
have to go to r = 64-256. (the paper uses 8 and when they used 16, it reduced perf ~8-16 should
be perfect for us, will have to play around with it.)
- alpha (a) adjusts how much the lora relies on the original parameters. low alpha -> more reliance on og params, high
alpha -> more emphasis on low-rank structure/regularization. usually use an alpha that is 2r.
- paper does lora on the self-attention module of transformer
