use candle_core::{IndexOp, Module, Tensor};
use candle_nn::{layer_norm, LayerNorm, LayerNormConfig, VarBuilder};
use crate::errors::HsError;
use crate::errors::HsError::InitModelError;
use crate::transformers::config::{EmbeddsReduction, TransformerModelConfig};
use crate::transformers::layers::{GenEmbeddLayer, TransformerEncoderLayer};

/// Generic struct that represents a Transformer model
///
/// Params:
///
/// [`A: GenEmbeddLayer + Module`]: Layer to produce embeddings (text, image, audio)
/// Type alias are available: VisionTransformer, TextTransformer, AudioTransformer
///
/// [`B: TransformerEncoderLayer + Module`]: Transformer encoder block to implement layer
/// EX: TransformerResBlock
///
#[derive(Debug)]
pub struct TransformerModelGen<A: GenEmbeddLayer + Module, B: TransformerEncoderLayer + Module> {
    embeddings: A,
    permutation: Option<Vec<usize>>,
    encoder: Vec<B>,
    reduction: Option<EmbeddsReduction>,
    projection: Tensor,
    ln_pre: Option<LayerNorm>,
    ln_post: Option<LayerNorm>

}

impl <A: GenEmbeddLayer + Module, B: TransformerEncoderLayer + Module> TransformerModelGen<A, B> {
    pub fn new<C: TransformerModelConfig>(vb: VarBuilder, config: &C) -> Result<Self, HsError> {
        let projection = vb.get((config.get_hidden_size(), config.get_embeddings_size()), config.get_projection_label()).map_err(InitModelError)?;
        let permutation = config.get_permutation();
        let mut encoder = Vec::new();
        let vb_layers = vb.pp(config.get_layers_label());
        for num_layer in 0..config.get_num_layers() {
            encoder.push(B::new(vb_layers.pp(num_layer.to_string()), &config.get_transformer_layer_config())?);
        }
        let reduction = config.get_reduction();
        let ln_pre = if let Some((label, dim)) = config.get_ln_pre_config() {
            Some(layer_norm(dim, LayerNormConfig::default(), vb.pp(label)).map_err(InitModelError)?)
        } else { None };

        let ln_post = if let Some((label, dim)) = config.get_ln_post_config() {
            Some(layer_norm(dim, LayerNormConfig::default(), vb.pp(label)).map_err(InitModelError)?)
        } else { None };

        let embeddings = A::new(vb, config)?; // borrows vb, has to be the last one


        Ok(Self {
            embeddings,
            permutation,
            encoder,
            reduction,
            projection,
            ln_pre,
            ln_post
        })
    }
}

impl <A: GenEmbeddLayer + Module, B: TransformerEncoderLayer + Module> Module for TransformerModelGen<A, B> {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        //[batch_size, num_patches, hidden_size]
        let mut xs = xs.apply(&self.embeddings)?;
        // class & positional
        // pre norm
        if let Some(ln_pre) = &self.ln_pre {
            xs = xs.apply(ln_pre)?;
        }

        // Permutation before & after transformer encoder will be optional
        if let Some(permutation) = &self.permutation {
            xs = xs.permute(permutation.as_slice())?;
            // [seq_len, batch_size, embedd_dim] with permutation (1, 0, 2)
        }

        // Apply all layers (1 layer)
        // (from Attention is all you need)
        // The encoder is composed of a stack of N identical layers. Each layer has two
        // sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. We employ a residual connection [11] around each of
        // the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is
        // LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
        // itself
        for encoder_layer in self.encoder.iter() {
            xs = xs.apply(encoder_layer)?;
            // [seq_len,  batch_size, embedd_dim] with B: TransformerResBlock & requires permutation (1, 0, 2)
        }

        if let Some(permutation) = &self.permutation {
            xs = xs.permute(permutation.as_slice())?;
            // [batch_size, seq_len, embedd_dim] with permutation (1, 0, 2)
        }


        if let Some(reduction) = &self.reduction {
            match reduction {
                // GlobalAveragePooling1D
                EmbeddsReduction::Mean => xs = xs.mean_keepdim(1)?,  // [batch_size, 1, embedd_dim]
                // Select 0 -> Clip ViT uses this
                EmbeddsReduction::Zero => xs =  xs.i((..,0,..))?, // [batch_size, 1, embedd_dim]
            }
        }
        //[batch_size, num_patches, hidden_size] if reduction has not been applied
        //[batch_size, 1, hidden_size] if reduction has been applied -> todo! check this, because we need [batch_size, embedd_dim]

        // post norm
        if let Some(ln_post) = &self.ln_post {
            xs = xs.apply(ln_post)?;
        }

        // projection: xs @ projection: [batch_size, embedd_dim] @ [embedd_dim, final_embedd_dim] = [batch_size, final_embedd_dim]
        xs = xs.matmul(&self.projection)?;
        Ok(xs)

    }
}