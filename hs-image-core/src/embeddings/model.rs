use hs_core::transformers::layers::TransformerResBlock;
use hs_core::transformers::model::TransformerModelGen;
use crate::embeddings::layers::VisionEmbedLayer;

pub type VisionTransformer = TransformerModelGen<VisionEmbedLayer, TransformerResBlock>;


#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use candle_core::{Device, DType, Module, Tensor};
    use candle_nn::{VarBuilder};
    use hs_core::transformers::config::{EmbeddsReduction, MlpLayerConfig, MsaLayerConfig, TransformerLayerConfig, VisionTransformerModelConfig};
    use hs_core::transformers::config::Activation::QuickGelu;

    use super::*;

    #[test]
    fn vision_transformer() {
        let channels = 3;
        let hidden_size = 768;
        let patch_size = 32;
        let num_patches = 49;
        let embeddings_size = 512;
        let num_layers = 3;
        let mut ts: HashMap<String, Tensor> = HashMap::new();
        ts.insert(String::from("conv.weight"), Tensor::ones((hidden_size, channels, patch_size, patch_size), DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("positional_embedding"), Tensor::ones((num_patches + 1, hidden_size), DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("class_embedding"), Tensor::ones(hidden_size, DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("proj"), Tensor::ones((hidden_size, embeddings_size), DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("ln_pre.weight"), Tensor::ones((hidden_size, hidden_size), DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("ln_pre.bias"), Tensor::ones(hidden_size, DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("ln_post.weight"), Tensor::ones((hidden_size, hidden_size), DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("ln_post.bias"), Tensor::ones(hidden_size, DType::F32, &Device::Cpu).unwrap());



        for num_layer in 0..num_layers {
            ts.insert(format!("transformer.resblocks.{}.ln_1.weight", num_layer), Tensor::ones(768, DType::F32, &Device::Cpu).unwrap());
            ts.insert(format!("transformer.resblocks.{}.ln_1.bias", num_layer), Tensor::ones(768, DType::F32, &Device::Cpu).unwrap());
            ts.insert(format!("transformer.resblocks.{}.ln_2.weight", num_layer), Tensor::ones(768, DType::F32, &Device::Cpu).unwrap());
            ts.insert(format!("transformer.resblocks.{}.ln_2.bias", num_layer), Tensor::ones(768, DType::F32, &Device::Cpu).unwrap());

            ts.insert(format!("transformer.resblocks.{}.mlp.c_fc.weight", num_layer), Tensor::ones((3072, 768), DType::F32, &Device::Cpu).unwrap());
            ts.insert(format!("transformer.resblocks.{}.mlp.c_fc.bias", num_layer), Tensor::ones(3072, DType::F32, &Device::Cpu).unwrap());
            ts.insert(format!("transformer.resblocks.{}.mlp.c_proj.weight", num_layer), Tensor::ones((768, 3072), DType::F32, &Device::Cpu).unwrap());
            ts.insert(format!("transformer.resblocks.{}.mlp.c_proj.bias", num_layer), Tensor::ones(768, DType::F32, &Device::Cpu).unwrap());

            ts.insert(format!("transformer.resblocks.{}.attn.in_proj.weight", num_layer), Tensor::ones((2304, 768), DType::F32, &Device::Cpu).unwrap());
            ts.insert(format!("transformer.resblocks.{}.attn.in_proj.bias", num_layer), Tensor::ones(2304, DType::F32, &Device::Cpu).unwrap());
            ts.insert(format!("transformer.resblocks.{}.attn.out_proj.weight", num_layer), Tensor::ones((768, 768), DType::F32, &Device::Cpu).unwrap());
            ts.insert(format!("transformer.resblocks.{}.attn.out_proj.bias", num_layer), Tensor::ones(768, DType::F32, &Device::Cpu).unwrap());
        }

        let vb = VarBuilder::from_tensors(ts, DType::F32, &Device::Cpu);

        let config = VisionTransformerModelConfig {
            input_size: 225,
            channels: 3,
            patch_size,
            hidden_size,
            num_patches,
            patch_embedding_label: String::from("conv"),
            positional_embedding_label: String::from("positional_embedding"),
            class_embedding_label: String::from("class_embedding"),
            projection_label: String::from("proj"),
            permutation: Some(Vec::from(&[1, 0 ,2])),
            embeddings_size,
            num_layers,
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
                    interm_size: 3072,
                    activation: Some(QuickGelu),
                    c_fc_label: String::from("c_fc"),
                    c_proj_label: String::from("c_proj"),
                },
                msa_layer: MsaLayerConfig {
                    embed_dim: hidden_size,
                    head_dim: 64,
                    num_patches: 49,
                    num_heads: 12,
                    interm_size: 2304,
                    in_proj_label: Some(String::from("in_proj")),
                    q_label: None,
                    k_label: None,
                    v_label: None,
                    out_proj_label: String::from("out_proj"),
                },
            },
            ln_pre_config: Some((String::from("ln_pre"), hidden_size, hidden_size)),
            ln_post_config: Some((String::from("ln_post"), hidden_size, hidden_size)),
        };

        let vision_transformer = VisionTransformer::new(vb, &config).unwrap();



        let batch_size = 1;
        let img_size = 224;
        let input: Tensor = Tensor::ones((batch_size, channels, img_size, img_size), DType::F32, &Device::Cpu).unwrap();

        let output: Tensor = vision_transformer.forward(&input).unwrap();

        assert_eq!(output.dims(), &[batch_size, embeddings_size]);
    }
}