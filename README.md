# Handspring
Open Source Multimodal Embeddings in Rust

:construction_worker_man: Work in progress


### Goals
- Access to state-of-the-art multimodal embedding models
- Only inference
- High performance on CPU
- Cheap & easy to scale
- Flexible integration

### Features

| Status      | Model                | Feature             | Embeddings  |
|-------------|----------------------|---------------------|-------------|
| Development | OpenAI/Clip-ViT-B-32 | clip_vit_b_32_image | Image       |
| Next        | OpenAI/Clip-ViT-B-32 | clip_vit_b_32_text  | Text        |
| -           | OpenAI/Clip-ViT-B-32 | clip_vit_b_32       | Image, Text |
| -           | ImageBind            |                     |             |
|             |                      |                     |             |


### Strategies
- Embed models config & safetensors files:
  - No need to handle extra files
- Reduce component size
  - Each feature will only be responsible for generating embeddings for 1 modality for 1 model
  - Extra feature for the full model (low priority)
  - Split safetensors files into modalities (todo! improve code to automate this)


### Docs & Examples
todo!

### Roadmap
- First Clip Model
- Output & Performance evaluation
- Add ImageBind Model
- Examples in Rust
- Examples integration with other languages
- Example [Weaviate custom module](https://weaviate.io/developers/weaviate/modules/other-modules/custom-modules)
- Expand models catalog

### Integrations with other programming languages
todo!

- Elixir: maybe using [Rustler](https://github.com/rusterlium/rustler)
- JVM langs: maybe JNI
- Python: [PyO3](https://github.com/PyO3/pyo3)



### Community Server
todo!


## Resources

### Dev Resources
- Rust basics
  - [Rust book](https://doc.rust-lang.org/book/title-page.html)
  - [log](https://docs.rs/log/latest/log/)
  - [serde](https://docs.rs/serde/latest/serde/)
  - [thiserror](https://docs.rs/thiserror/latest/thiserror/)
  - [rust-embed](https://github.com/pyrossh/rust-embed)
- Rust ML 
  - [HuggingFace Candle Framework](https://github.com/huggingface/candle)
- []()

### Papers & Models
- [Attention is all you Need](https://arxiv.org/abs/1706.03762)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [OpenAI CLIP](https://openai.com/research/clip)
- [Meta ImageBind](https://imagebind.metademolab.com/)