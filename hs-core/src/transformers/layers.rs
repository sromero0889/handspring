
// todo!
// Pending following blocks inside TransformerResBlock:

// MultiHeadSelfAttentionLayer (MSP)
// This layer concatenates all the attention outputs linearly to the right dimensions. The many attention heads help train local and global dependencies in an image.

// MultiLayerPerceptronsLayer (MLP)
// This layer contains a two-layer with Gaussian Error Linear Unit (GELU)


// LayerNorm -> available in candle




use candle_core::{Module, Tensor, IndexOp};
use candle_nn::VarBuilder;
use crate::errors::HsError;
use crate::errors::HsError::InitModelError;
use crate::transformers::config::{EmbeddsReduction, TransformerModelConfig};



pub trait TransformerEncoderLayer {
    fn new<C: TransformerModelConfig>(vb: VarBuilder, config: &C) -> Result<Self, HsError> where Self: Sized;
}

#[derive(Debug)]
pub struct TransformerResBlock {

}

impl TransformerEncoderLayer for TransformerResBlock {
    fn new<C: TransformerModelConfig>(vb: VarBuilder, config: &C) -> Result<Self, HsError> where Self: Sized {
        todo!()
    }
}

impl Module for TransformerResBlock {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        todo!()
    }
}

pub trait GenEmbeddLayer {
    fn new<C: TransformerModelConfig>(vb: VarBuilder, config: &C) -> Result<Self, HsError> where Self: Sized;
}


/// Generic struct that represents a Transformer model
///
/// Params:
///
/// [`A: GenEmbeddLayer + Module`]: Layer to produce embeddings (text, image, audio)
/// Type alias are available: VisionTransformer, TextTransformer, AudioTransformer
///
/// [`A: TransformerEncoderLayer + Module`]: Transformer encoder block
///
///
/// # Examples
///
/// ```
/// todo!()
/// ```
#[derive(Debug)]
pub struct TransformerModelGen<A: GenEmbeddLayer + Module, B: TransformerEncoderLayer + Module> {
    embeddings: A,
    permutation: Option<Vec<usize>>,
    encoder: Vec<B>,
    reduction: Option<EmbeddsReduction>,
    projection: Tensor,

}

impl <A: GenEmbeddLayer + Module, B: TransformerEncoderLayer + Module> TransformerModelGen<A, B> {
    pub fn new<C: TransformerModelConfig>(vb: VarBuilder, config: &C) -> Result<Self, HsError> {
        let projection = vb.get((config.get_hidden_size(), config.get_embeddings_size()), config.get_projection_label()).map_err(InitModelError)?;
        let permutation = config.get_permutation();
        let mut encoder = Vec::new();
        let vb_layers = vb.pp(config.get_layers_label());
        for num_layer in 0..config.get_num_layers() {
            encoder.push(B::new(vb_layers.pp(num_layer.to_string()), config)?);
        }
        let reduction = config.get_reduction();
        let embeddings = A::new(vb, config)?; // borrows vb, has to be the last one


        Ok(Self {
            embeddings,
            permutation,
            encoder,
            reduction,
            projection
        })
    }
}

impl <A: GenEmbeddLayer + Module, B: TransformerEncoderLayer + Module> Module for TransformerModelGen<A, B> {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        //[batch_size, num_patches, hidden_size]
        let mut xs = xs.apply(&self.embeddings)?;
        // class & positional
        // pre norm

        // Permutation before & after transformer encoder will be optional
        if let Some(permutation) = &self.permutation {
            xs = xs.permute(permutation.as_slice())?;
        }

        // Apply all layers (1 layer)
        // (from Attention is all you need)
        // The encoder is composed of a stack of N identical layers. Each layer has two
        // sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. We employ a residual connection [11] around each of
        // the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is
        // LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
        // itself
        for encoder_layer in self.encoder.iter() {
            xs = xs;
        }

        if let Some(permutation) = &self.permutation {
            xs = xs.permute(permutation.as_slice())?;
        }


        if let Some(reduction) = &self.reduction {
            match reduction {
                // GlobalAveragePooling1D
                EmbeddsReduction::Mean => xs = xs.mean_keepdim(1)?,
                // Select 0
                EmbeddsReduction::Zero => xs =  xs.i((..,0,..))?,
            }
        }

        // post norm

        // projection xs @ projection
        xs = xs.matmul(&self.projection)?;
        Ok(xs)
        //[batch_size, num_patches, hidden_size] if reduction has not been applied
        //[batch_size, 1, hidden_size] if reduction has been applied
    }
}