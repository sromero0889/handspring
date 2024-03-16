use serde::{Deserialize, Serialize};
use hs_audio_core::preprocessing::config::AudioPreprocessingConfig;
use hs_core::transformers::config::{AudioTransformerModelConfig, EmbeddingsMultimodalConfig, MultimodalConfig, TextTransformerModelConfig, VisionTransformerModelConfig};
use hs_image_core::preprocessing::config::ImagePreprocessingConfig;
use hs_text_core::preprocessing::config::TextPreprocessingConfig;

#[derive(Serialize, Deserialize, Debug)]
pub struct ModelDescriptor {
    pub preprocessing: MultimodalConfig<ImagePreprocessingConfig,TextPreprocessingConfig, AudioPreprocessingConfig>,
    pub embeddings: EmbeddingsMultimodalConfig<VisionTransformerModelConfig,TextTransformerModelConfig, AudioTransformerModelConfig>,
}