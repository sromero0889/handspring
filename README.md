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


### Performance tests
#### 19-Mar-2024
Basic test to get an idea of the current performance
CPU, same environment

**Clip python, JIT**

```
device="cpu"
model_name="ViT-B/32"
model, transform = clip.load(model_name, device=device, jit=True)

input_data_b = torch.ones((5, 3, 224, 224), dtype=torch.float32, device=device)
out = model.encode_image(input_data_b)
```

Results
```
Time elapsed in build_model() is: 1.6573503890000003
Time elapsed in forward() is: 1.101630814
```

**Rust, Clip_vit_b_32_image**
```
use hs_mm_embeddings::clip_vit_b_32_image;
fn main() {
    let start = Instant::now();
    let image_model = clip_vit_b_32_image::model::build_model().unwrap();

    let duration = start.elapsed();
    println!("Time elapsed in build_model() is: {:?}", duration);
    
    let input_img_batch = Tensor::ones((5, 3, 224, 224), DType::F32, &Device::Cpu).unwrap();

    let start = Instant::now();

    let output = image_model.forward(&input_img_batch).unwrap();
    
    let duration = start.elapsed();
    println!("Time elapsed in forward() is: {:?}", duration);
```

Results

```
Time elapsed in build_model() is: 483.20213ms
Time elapsed in forward() is: 1.495299234s

```

**Testing different batch size**
- size 1:
  - Python: 0.5376610519999998
  - Rust: 282.812172ms
- size 2:
  - Python: 0.6142103929999996
  - Rust: 941.727512ms
- 


**Observations**
- Time generating embeddings still too high when batch_size > 1
- With batch_size = 1 is faster
- Max difference between output tensors: [8.5831e-6] JIT=False & [7.6294e-6] JIT=True
- Rust executable size (release): 183.7 MB

todo!: check unnecessary copies or redundant ops

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