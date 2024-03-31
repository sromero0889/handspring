use candle_core::{Device, DType};
use candle_nn::VarBuilder;
use log::error;
use rust_embed::RustEmbed;
use hs_core::errors::HsError;
use hs_core::errors::HsError::{JsonMapError, InitModelError, ModelAssetsNotFound, SafetensorsMapError, Other};
// use hs_image_core::embeddings::model::VisionTransformer;
use crate::config::ModelDescriptor;


pub fn build_model() ->  Result<(), HsError> {
    let (_model, _config) = load_model_data()?;
    todo!()
}


#[derive(RustEmbed)]
#[folder = "models/clip_vit_b_32/full/"]
struct ModelAssets;

// Loads Safetensors & Config Files
// Those file have been embedded using RustEmbed during compilation (required model feature: clip_vit_b_32)
fn load_model_data<'a>() ->  Result<(VarBuilder<'a>, ModelDescriptor), HsError> {

    let model = &ModelAssets::get("model.safetensors");
    let config = &ModelAssets::get("config.json");

    match (model, config) {
        (Some(model), Some(config)) => {
            let model = VarBuilder::from_buffered_safetensors(model.data.to_vec(), DType::F32, &Device::Cpu)
                .map_err(SafetensorsMapError)?;
            let config: ModelDescriptor =  serde_json::from_slice(&config.data.to_vec()).map_err(JsonMapError)?;
            Ok((model, config))
        }
        _ => {
            let error = ModelAssetsNotFound("clip_v it_b_32");
            error!("{}", error);
            Err(error)
        }
    }
}

