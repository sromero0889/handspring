use candle_core::{Module, Tensor};
use candle_nn::VarBuilder;
use hs_core::errors::HsError;
use hs_core::errors::HsError::InitModelError;
use hs_core::transformers::config::TransformerModelConfig;
use hs_core::transformers::layers::GenEmbeddLayer;

#[derive(Debug)]
pub struct PatchEmbeddingLayerConfig {
    pub channels: usize,
    pub hidden_size: usize,
    pub patch_size: usize
}


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
            .permute((0,2,1))
            //[batch_size, num_patches, hidden_size]
    }
}


pub struct VisionEmbedLayer {
    hidden_size: usize,
    patch_embedding_layer: PatchEmbeddingLayer,
    // position_ids: Tensor,
    positional_embedding: Tensor,
    class_embedding: Tensor
}

impl GenEmbeddLayer for VisionEmbedLayer {
    fn new<C: TransformerModelConfig>(vb: VarBuilder, config: &C) -> Result<Self, HsError> where Self: Sized {
        let config = config.get_vision()?;
        let vb_patch_embedding_layer = vb.pp(config.patch_embedding_label.as_str());
        let config_patch_embedding_layer = PatchEmbeddingLayerConfig {
            channels: config.channels,
            hidden_size: config.hidden_size,
            patch_size: config.patch_size,
        };
        let patch_embedding_layer = PatchEmbeddingLayer::new(
            vb_patch_embedding_layer,
            &config_patch_embedding_layer
        )?;

        // let position_ids = Tensor::arange(0u32, (config.num_patches + 1) as u32, &Device::Cpu).map_err(InitModelError)?.unsqueeze(0).map_err(InitModelError)?;
        let positional_embedding = vb.get((config.num_patches + 1, config.hidden_size), config.positional_embedding_label.as_str()).map_err(InitModelError)?;
        let class_embedding = vb.get( config.hidden_size, config.class_embedding_label.as_str()).map_err(InitModelError)?;
        Ok(Self {
            hidden_size: config.hidden_size,
            patch_embedding_layer,
            // position_ids,
            positional_embedding,
            class_embedding
        })
    }

}

impl Module for VisionEmbedLayer {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        //xs [batch_size, channel, input_size, input_size]
        let patch_embeds = xs.apply(&self.patch_embedding_layer)?;
        //patch_embeds [batch_size, num_patches, hidden_size]
        let class_embeds = self.class_embedding.expand((xs.dim(0)?, 1, self.hidden_size))?;
        //class_embeds [batch_size, 1, hidden_size]
        let embeddings = Tensor::cat(&[&class_embeds, &patch_embeds], 1)?;
        // embeddings [batch_size, num_patches + 1, hidden_size]
        embeddings.broadcast_add(&self.positional_embedding)
        //embeddings [batch_size, num_patches + 1, hidden_size]
    }
}





#[cfg(test)]
mod tests {
    use std::any::type_name_of_val;
    use std::collections::HashMap;
    use candle_core::{Device, DType, Tensor};
    use hs_core::transformers::config::{EmbeddsReduction, LayerNormConfig, MlpLayerConfig, MsaLayerConfig, TransformerLayerConfig, VisionTransformerModelConfig};
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

    #[test]
    fn vision_embedd_layer_ok() {
        let channels = 3;
        let hidden_size = 768;
        let patch_size = 32;
        let num_patches = 49;
        let mut ts: HashMap<String, Tensor> = HashMap::new();
        ts.insert(String::from("conv.weight"), Tensor::ones((hidden_size, channels, patch_size, patch_size), DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("positional_embedding"), Tensor::ones((num_patches + 1, hidden_size), DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("class_embedding"), Tensor::ones(hidden_size, DType::F32, &Device::Cpu).unwrap());
        let vb = VarBuilder::from_tensors(ts, DType::F32, &Device::Cpu);

        let config = VisionTransformerModelConfig {
            input_size: 225,
            channels: 3,
            patch_size,
            hidden_size,
            num_patches,
            embeddings_size: 512,
            patch_embedding_label: String::from("conv"),
            positional_embedding_label: String::from("positional_embedding"),
            class_embedding_label: String::from("class_embedding"),
            projection_label: String::from("proj"),
            permutation: Some(Vec::from(&[1, 0 ,2])),
            num_layers: 1,
            layers_label: String::from("transformer.resblocks"),
            reduction: Some(EmbeddsReduction::Zero),
            transformer_layer: TransformerLayerConfig {
                hidden_size,
                ln_1_label: String::from("ln_1"),
                ln_2_label: String::from("ln_2"),
                msa_label: String::from("attn"),
                mlp_label: String::from("mlp"),
                mlp_layer: MlpLayerConfig {
                    hidden_size,
                    interm_size: 0,
                    activation: None,
                    c_fc_label: String::from(""),
                    c_proj_label: String::from(""),
                },
                msa_layer: MsaLayerConfig {
                    embed_dim: 0,
                    head_dim: 0,
                    num_patches,
                    num_heads: 0,
                    interm_size: 0,
                    in_proj_label: None,
                    q_label: None,
                    k_label: None,
                    v_label: None,
                    out_proj_label: String::from(""),
                },
            },
            ln_pre_config: Some(LayerNormConfig {
                label: String::from("ln_pre"),
                in_dim: hidden_size,
                out_dim: hidden_size
            }),
            ln_post_config: Some(LayerNormConfig {
                label: String::from("ln_post"),
                in_dim: hidden_size,
                out_dim: hidden_size
            })
        };

        let vision_embedd_layer = VisionEmbedLayer::new(vb, &config).unwrap();


        let batch_size = 1;
        let img_size = 224;
        let input: Tensor = Tensor::ones((batch_size, channels, img_size, img_size), DType::F32, &Device::Cpu).unwrap();

        let output: Tensor = vision_embedd_layer.forward(&input).unwrap();

        assert_eq!(output.dims(), &[batch_size, num_patches + 1, hidden_size]);
    }

}