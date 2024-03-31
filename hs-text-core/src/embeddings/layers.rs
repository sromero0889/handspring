use candle_core::{Module, Tensor};
use candle_nn::VarBuilder;
use hs_core::errors::HsError;
use hs_core::transformers::config::{TransformerModelConfig};
use hs_core::transformers::layers::GenEmbeddLayer;

pub struct TextEmbedLayer {}
impl GenEmbeddLayer for TextEmbedLayer {

    fn new<C: TransformerModelConfig>(vb: VarBuilder, config: &C) -> Result<Self, HsError>  where Self: Sized {
        unimplemented!()
    }
}

impl Module for TextEmbedLayer {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        unimplemented!()
    }
}

