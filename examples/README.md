## Examples

## CLIP

### CLIP ViT-B-32
CPU inference
Model loaded: clip_vit_b_32
Features needed: clip_vit_b_32_image & clip_vit_b_32_text

Generates embeddings from an image and a text and compares them using cosine similarity.

Run example
```bash
cargo run --example clip_vit_b_32
```

#### CLIP ViT-B-32 Image Embeddings
CPU inference
Model loaded: clip_vit_b_32 (partial weights: only vision)
Features needed: clip_vit_b_32_image

Generates embeddings from an image.

Run example
```bash
cargo run --example clip_vit_b_32_image
```

#### CLIP ViT-B-32 Text Embeddings
CPU inference
Model loaded: clip_vit_b_32 (partial weights: only text)
Features needed: clip_vit_b_32_text

Generates embeddings from a text.

Run example
```bash
cargo run --example clip_vit_b_32_text
```

