

#[derive(thiserror::Error, Debug)]
pub enum HsError {
    #[error("Assets Not found, Model {0}")]
    ModelAssetsNotFound(&'static str),
    #[error("Init Model {1} failed: {0}")]
    InitModelError(#[from] candle_core::Error),
    // #[error("Init Model {1} failed: {0}")]
    // InitModelError(&'static str, &'static str),
    #[error("Error mapping json file")]
    JsonMapError(#[from] serde_json::Error),
    #[error("Error mapping safetensors file")]
    SafetensorsMapError(#[from] candle_core::Error),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}