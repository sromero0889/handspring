use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct PatchEmbeddingLayerConfig {
    pub channels: usize,
    pub hidden_size: usize,
    pub patch_size: usize
}