use candle_core::Module;
use hs_core::transformers::layers::TransformerResBlock;
use hs_core::transformers::model::TransformerModelGen;
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