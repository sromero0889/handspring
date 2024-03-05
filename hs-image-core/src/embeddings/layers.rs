use candle_core::{Module, Tensor};
use candle_nn::VarBuilder;
use hs_core::errors::HsError;
use hs_core::errors::HsError::InitModelError;
use crate::embeddings::config::PatchEmbeddingLayerConfig;

#[derive(Debug)]
pub struct PatchEmbeddingLayer {
    conv2d: candle_nn::Conv2d,
}

impl PatchEmbeddingLayer {
    pub fn new(vb: VarBuilder, config: &PatchEmbeddingLayerConfig) -> Result<Self, HsError> {

        Ok(Self {
            conv2d: candle_nn::conv2d_no_bias(
                config.channels,
                config.hidden_size,
                config.patch_size,
                candle_nn::Conv2dConfig {
                    padding: 0,
                    stride: config.patch_size,
                    dilation: 1,
                    groups: 1,
                },
                vb
            ).map_err(InitModelError)?
        })
    }
}

impl Module for PatchEmbeddingLayer {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        //[batch_size, channels, img_size, img_size]
        xs.apply(&self.conv2d)?
            //[batch_size, hidden_size, sqrt(num_patches), sqrt(num_patches)]
            .flatten_from(2)?
            //[batch_size, hidden_size, num_patches]
            .transpose(1,2)
            //[batch_size, num_patches, hidden_size]
    }
}

#[cfg(test)]
mod tests {
    use std::any::type_name_of_val;
    use std::collections::HashMap;
    use candle_core::{Device, DType, Tensor};
    use super::*;

    #[test]
    fn patch_embedding_layer_ok() {
        let channels = 3;
        let hidden_size = 768;
        let patch_size = 32;
        let mut ts: HashMap<String, Tensor> = HashMap::new();
        ts.insert(String::from("conv.weight"), Tensor::ones((hidden_size, channels, patch_size, patch_size), DType::F32, &Device::Cpu).unwrap());
        let vb = VarBuilder::from_tensors(ts, DType::F32, &Device::Cpu);

        let config: PatchEmbeddingLayerConfig = PatchEmbeddingLayerConfig {
            channels,
            hidden_size,
            patch_size,
        };
        let patch_embed_layer = PatchEmbeddingLayer::new(vb.pp("conv"), &config).unwrap();

        let batch_size = 1;
        let img_size = 224;
        let input: Tensor = Tensor::ones((batch_size, channels, img_size, img_size), DType::F32, &Device::Cpu).unwrap();

        let output: Tensor = patch_embed_layer.forward(&input).unwrap();

        let num_patches = img_size / patch_size;
        let num_patches = num_patches * num_patches;
        assert_eq!(output.dims(), &[batch_size, num_patches, hidden_size]);

    }

    #[test]
    fn patch_embedding_layer_ko_init_layer() {
        let channels = 3;
        let hidden_size = 768;
        let patch_size = 32;
        let mut ts: HashMap<String, Tensor> = HashMap::new();
        ts.insert(String::from("conv.wrong_key"), Tensor::ones((hidden_size, channels, patch_size, patch_size), DType::F32, &Device::Cpu).unwrap());
        let vb = VarBuilder::from_tensors(ts, DType::F32, &Device::Cpu);

        let config: PatchEmbeddingLayerConfig = PatchEmbeddingLayerConfig {
            channels,
            hidden_size,
            patch_size,
        };
        let patch_embed_layer = PatchEmbeddingLayer::new(vb.pp("conv"), &config);
        match patch_embed_layer {
            Err(InitModelError(_err)) => assert!(true),
            _ => assert!(false)
        }
    }

    #[test]
    fn patch_embedding_layer_ko_input_shape() {
        let channels = 3;
        let hidden_size = 768;
        let patch_size = 32;
        let mut ts: HashMap<String, Tensor> = HashMap::new();
        ts.insert(String::from("conv.weight"), Tensor::ones((hidden_size, channels, patch_size, patch_size), DType::F32, &Device::Cpu).unwrap());
        let vb = VarBuilder::from_tensors(ts, DType::F32, &Device::Cpu);

        let config: PatchEmbeddingLayerConfig = PatchEmbeddingLayerConfig {
            channels,
            hidden_size,
            patch_size,
        };
        let patch_embed_layer = PatchEmbeddingLayer::new(vb.pp("conv"), &config).unwrap();

        let batch_size = 1;
        let input: Tensor = Tensor::ones((batch_size, 1), DType::F32, &Device::Cpu).unwrap();

        let output = patch_embed_layer.forward(&input);
        match output {
            Err(err) => assert_eq!(type_name_of_val(&err), "candle_core::error::Error"),
            _ => assert!(false)
        }
    }
}