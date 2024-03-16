use serde::{Deserialize, Serialize};
use crate::errors::HsError;


#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MlpLayerConfig {
    pub hidden_size: usize,
    pub interm_size: usize,
    pub activation: Option<Activation>,
    pub c_fc_label: &'static str,
    pub c_proj_label: &'static str
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MsaLayerConfig {
    pub embed_dim: usize,
    pub head_dim: usize,
    pub num_patches: usize,
    pub num_heads: usize,
    pub interm_size: usize,
    pub in_proj_label: Option<&'static str>,
    pub q_label: Option<&'static str>,
    pub k_label: Option<&'static str>,
    pub v_label: Option<&'static str>,
    pub out_proj_label: &'static str
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Activation {
    QuickGelu,
    Gelu
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum EmbeddsReduction {
    Mean,
    Zero
}

pub trait GenEmbeddLayerConfig {}

pub trait TransformerModelConfig {
    fn get_vision(&self) -> Result<&VisionTransformerModelConfig, HsError>;
    fn get_text(&self) -> Result<&TextTransformerModelConfig, HsError>;
    fn get_audio(&self) -> Result<&AudioTransformerModelConfig, HsError>;
    fn get_hidden_size(&self) -> usize;
    fn get_embeddings_size(&self) -> usize;
    fn get_projection_label(&self) -> &'static str;
    fn get_permutation(&self) -> Option<Vec<usize>>;
    fn get_reduction(&self) -> Option<EmbeddsReduction>;
    fn get_num_layers(&self) -> usize;
    fn get_layers_label(&self) -> &'static str;
    fn get_transformer_layer_config(&self) -> &TransformerLayerConfig;
    fn get_ln_pre_config(&self) -> Option<(&'static str, usize, usize)>;
    fn get_ln_post_config(&self) -> Option<(&'static str, usize, usize)>;
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TransformerLayerConfig {
    pub hidden_size: usize,
    pub ln_1_label: &'static str,
    pub ln_2_label: &'static str,
    pub msa_label: &'static str,
    pub mlp_label: &'static str,
    pub mlp_layer: MlpLayerConfig,
    pub msa_layer: MsaLayerConfig,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VisionTransformerModelConfig {
    pub input_size: usize,
    pub channels: usize,
    pub patch_size: usize,
    pub hidden_size: usize,
    pub num_patches: usize,
    pub embeddings_size: usize,
    pub num_layers: usize,
    pub layers_label: &'static str,
    pub patch_embedding_label: &'static str,
    pub positional_embedding_label: &'static str,
    pub class_embedding_label: &'static str,
    pub projection_label: &'static str,
    pub permutation: Option<Vec<usize>>,
    pub reduction: Option<EmbeddsReduction>,
    pub transformer_layer: TransformerLayerConfig,
    pub ln_pre_config: Option<(&'static str, usize, usize)>,
    pub ln_post_config: Option<(&'static str, usize, usize)>
}

impl TransformerModelConfig for VisionTransformerModelConfig {
    fn get_vision(&self) -> Result<&VisionTransformerModelConfig, HsError> {
        Ok(&self)
    }

    fn get_text(&self) -> Result<&TextTransformerModelConfig, HsError> {
        unimplemented!()
    }

    fn get_audio(&self) -> Result<&AudioTransformerModelConfig, HsError> {
        unimplemented!()
    }

    fn get_hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn get_embeddings_size(&self) -> usize {
        self.embeddings_size
    }

    fn get_projection_label(&self) -> &'static str {
        self.projection_label
    }

    fn get_permutation(&self) -> Option<Vec<usize>> {
        self.permutation.clone()
    }

    fn get_reduction(&self) -> Option<EmbeddsReduction> {
        self.reduction.clone()
    }

    fn get_num_layers(&self) -> usize {
        self.num_layers
    }

    fn get_layers_label(&self) -> &'static str {
        self.layers_label
    }

    fn get_transformer_layer_config(&self) -> &TransformerLayerConfig {
        &self.transformer_layer
    }

    fn get_ln_pre_config(&self) -> Option<(&'static str, usize, usize)> {
        self.ln_pre_config
    }

    fn get_ln_post_config(&self) -> Option<(&'static str, usize, usize)> {
        self.ln_post_config
    }
}



#[derive(Serialize, Deserialize, Debug)]
pub struct TextTransformerModelConfig {
    pub hidden_size: usize,
    pub embeddings_size: usize,
    pub num_layers: usize,
    pub layers_label: &'static str,
    pub projection_label: &'static str,
    pub permutation: Option<Vec<usize>>,
    pub transformer_layer: TransformerLayerConfig,
    pub ln_pre_config: Option<(&'static str, usize, usize)>,
    pub ln_post_config: Option<(&'static str, usize, usize)>
}

impl TransformerModelConfig for TextTransformerModelConfig {
    fn get_vision(&self) -> Result<&VisionTransformerModelConfig, HsError> {
        unimplemented!()
    }

    fn get_text(&self) -> Result<&TextTransformerModelConfig, HsError> {
        Ok(&self)
    }

    fn get_audio(&self) -> Result<&AudioTransformerModelConfig, HsError> {
        unimplemented!()
    }

    fn get_hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn get_embeddings_size(&self) -> usize {
        self.embeddings_size
    }

    fn get_projection_label(&self) -> &'static str {
        self.projection_label
    }

    fn get_permutation(&self) -> Option<Vec<usize>> {
        self.permutation.clone()
    }

    fn get_reduction(&self) -> Option<EmbeddsReduction> {
        None
    }

    fn get_num_layers(&self) -> usize {
        self.num_layers
    }

    fn get_layers_label(&self) -> &'static str {
        self.layers_label
    }

    fn get_transformer_layer_config(&self) -> &TransformerLayerConfig {
        &self.transformer_layer
    }

    fn get_ln_pre_config(&self) -> Option<(&'static str, usize, usize)> {
        self.ln_pre_config
    }

    fn get_ln_post_config(&self) -> Option<(&'static str, usize, usize)> {
        self.ln_post_config
    }
}


#[derive(Serialize, Deserialize, Debug)]
pub struct AudioTransformerModelConfig {

}

impl TransformerModelConfig for AudioTransformerModelConfig {
    fn get_vision(&self) -> Result<&VisionTransformerModelConfig, HsError> {
        unimplemented!()
    }

    fn get_text(&self) -> Result<&TextTransformerModelConfig, HsError> {
        unimplemented!()
    }

    fn get_audio(&self) -> Result<&AudioTransformerModelConfig, HsError> {
        Ok(&self)
    }

    fn get_hidden_size(&self) -> usize {
        unimplemented!()
    }

    fn get_embeddings_size(&self) -> usize {
        unimplemented!()
    }

    fn get_projection_label(&self) -> &'static str {
        unimplemented!()
    }

    fn get_permutation(&self) -> Option<Vec<usize>> {
        unimplemented!()
    }

    fn get_reduction(&self) -> Option<EmbeddsReduction> {
        unimplemented!()
    }

    fn get_num_layers(&self) -> usize {
        unimplemented!()
    }

    fn get_layers_label(&self) -> &'static str {
        unimplemented!()
    }

    fn get_transformer_layer_config(&self) -> &TransformerLayerConfig {
        unimplemented!()
    }

    fn get_ln_pre_config(&self) -> Option<(&'static str, usize, usize)> {
        unimplemented!()
    }

    fn get_ln_post_config(&self) -> Option<(&'static str, usize, usize)> {
        unimplemented!()
    }
}


#[derive(Serialize, Deserialize, Debug)]
pub struct MultimodalConfig<I,T,A> {
    image: Option<I>,
    text: Option<T>,
    audio: Option<A>,
}
