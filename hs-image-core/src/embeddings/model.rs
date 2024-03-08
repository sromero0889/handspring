use candle_nn::Linear;
use hs_core::transformers::layers::{TransformerModelGen, TransformerResBlock};
use crate::embeddings::layers::VisionEmbedLayer;

pub type VisionTransformer = TransformerModelGen<VisionEmbedLayer, TransformerResBlock>;


#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use candle_core::{Device, DType, Tensor};
    use candle_nn::VarBuilder;
    use hs_core::transformers::config::{EmbeddsReduction, TransformerLayerConfig, VisionTransformerModelConfig};

    use super::*;

    #[test]
    fn vision_transformer() {
        let channels = 3;
        let hidden_size = 768;
        let patch_size = 32;
        let num_patches = 49;
        let embeddings_size = 512;
        let mut ts: HashMap<String, Tensor> = HashMap::new();
        ts.insert(String::from("conv.weight"), Tensor::ones((hidden_size, channels, patch_size, patch_size), DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("positional_embedding"), Tensor::ones((num_patches + 1, hidden_size), DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("class_embedding"), Tensor::ones((hidden_size), DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("proj"), Tensor::ones((hidden_size, embeddings_size), DType::F32, &Device::Cpu).unwrap());
        let vb = VarBuilder::from_tensors(ts, DType::F32, &Device::Cpu);

        let config = VisionTransformerModelConfig {
            input_size: 225,
            channels: 3,
            patch_size,
            hidden_size,
            num_patches,
            patch_embedding_label: "conv",
            positional_embedding_label: "positional_embedding",
            class_embedding_label: "class_embedding",
            projection_label: "proj",
            permutation: Some(Vec::from(&[1, 0 ,2])),
            embeddings_size,
            num_layers: 1,
            layers_label: "transformer.resblocks",
            reduction: Some(EmbeddsReduction::Zero),
            transformer_layer: TransformerLayerConfig {},
        };

        let vision_transformer = VisionTransformer::new(vb, &config).unwrap();



        let batch_size = 1;
        let img_size = 224;
        let input: Tensor = Tensor::ones((batch_size, channels, img_size, img_size), DType::F32, &Device::Cpu).unwrap();

        // let output: Tensor = vision_embedd_layer.forward(&input).unwrap();

        // assert_eq!(output.dims(), &[batch_size, num_patches + 1, hidden_size]);
    }
}