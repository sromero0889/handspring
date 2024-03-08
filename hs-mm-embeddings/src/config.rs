use serde::{Deserialize, Serialize};
use hs_audio_core::preprocessing::config::AudioPreprocessingConfig;
use hs_core::transformers::config::{AudioTransformerModelConfig, MultimodalConfig, TextTransformerModelConfig, VisionTransformerModelConfig};
use hs_image_core::preprocessing::config::ImagePreprocessingConfig;
use hs_text_core::preprocessing::config::TextPreprocessingConfig;

#[derive(Serialize, Deserialize, Debug)]
pub struct ModelDescriptor {
    preprocessing: MultimodalConfig<ImagePreprocessingConfig,TextPreprocessingConfig, AudioPreprocessingConfig>,
    embeddings: MultimodalConfig<VisionTransformerModelConfig,TextTransformerModelConfig, AudioTransformerModelConfig>,
}