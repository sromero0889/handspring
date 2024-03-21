use candle_core::{Device, DType};
use candle_nn::VarBuilder;
use log::{debug, error};
use rust_embed::RustEmbed;
use hs_core::errors::HsError;
use hs_core::errors::HsError::{JsonMapError, ModelAssetsNotFound, SafetensorsMapError, Other};
use hs_image_core::embeddings::model::VisionTransformer;
use crate::config::ModelDescriptor;


/// Returns a model that generates embeddings from images
///
/// # Examples
///
/// ```
/// use candle_core::{Device, DType, Module};
/// use hs_mm_embeddings::clip_vit_b_32_image;
///
/// // Load & preprocess your images instead
/// let batch_size = 1;
/// let channels = 3;
/// let img_size = 224;
/// let input_img_batch = candle_core::Tensor::ones((batch_size, channels, img_size, img_size), DType::F32, &Device::Cpu)?;
///
/// let image_model = clip_vit_b_32_image::model::build_model()?;
/// let output_batch = image_model.forward(&input_img_batch)?;
/// ```
///
pub fn build_model() ->  Result<VisionTransformer, HsError> {
    let (model, config) = load_model_data()?;
    debug!("{:?}",config);
    match &config.embeddings.image {
        Some(config) => VisionTransformer::new(model.pp("visual"), config),
        _ => Err(Other("Image model config not found or incorrect"))
    }
}


#[derive(RustEmbed)]
#[folder = "models/clip_vit_b_32/image/"]
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
            let error = ModelAssetsNotFound("clip_vit_b_32/image");
            error!("{}", error);
            Err(error)
        }
    }
}

