use serde::{Deserialize, Serialize};

// Todo
#[derive(Serialize, Deserialize, Debug)]
pub struct ImagePreprocessingConfig {
    size: usize
}
#[derive(Serialize, Deserialize, Debug)]
pub struct TextPreprocessingConfig {
    size: usize

}#[derive(Serialize, Deserialize, Debug)]
pub struct AudioPreprocessingConfig {
    size: usize
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ImageEmbeddingsConfig {
    size: usize
}
#[derive(Serialize, Deserialize, Debug)]
pub struct TextEmbeddingsConfig {
    size: usize

}#[derive(Serialize, Deserialize, Debug)]
pub struct AudioEmbeddingsConfig {
    size: usize
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MultimodalConfig<I,T,A> {
    image: Option<I>,
    text: Option<T>,
    audio: Option<A>,
}




#[derive(Serialize, Deserialize, Debug)]
pub struct ModelDescriptor {
    preprocessing: MultimodalConfig<ImagePreprocessingConfig,TextPreprocessingConfig, AudioPreprocessingConfig>,
    embeddings: MultimodalConfig<ImageEmbeddingsConfig,TextEmbeddingsConfig, AudioEmbeddingsConfig>,
}