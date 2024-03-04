use hs_mm_embeddings::clip_vit_b_32;
fn main() {
    env_logger::init();

    let _model = clip_vit_b_32::model::build_model().unwrap();
}
