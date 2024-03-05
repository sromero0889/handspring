

#[derive(thiserror::Error, Debug)]
pub enum HsError {
    #[error("Assets Not found, Model {0}")]
    ModelAssetsNotFound(&'static str),
    #[error("Init Model failed")]
    InitModelError(#[source] candle_core::Error),
    #[error("Error mapping json file")]
    JsonMapError(#[from] serde_json::Error),
    #[error("Error mapping safetensors file")]
    SafetensorsMapError(#[source] candle_core::Error),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}