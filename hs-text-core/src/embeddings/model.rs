use candle_core::Module;
use hs_core::transformers::layers::{TransformerModelGen, TransformerResBlock};
use crate::embeddings::layers::TextEmbedLayer;


pub type TextTransformer = TransformerModelGen<TextEmbedLayer, TransformerResBlock>;


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a() {
        assert!(true);
    }
}