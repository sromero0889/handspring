//! clip_vit_b_32 module
//! Requires feature clip_vit_b_32
//! Generate image-text embeddings using OpenAI CLIP ViT-B-32

#[cfg(feature = "clip_vit_b_32")]
pub mod clip_vit_b_32;


pub mod config;
#[cfg(feature = "clip_vit_b_32_image")]
pub mod clip_vit_b_32_image;

