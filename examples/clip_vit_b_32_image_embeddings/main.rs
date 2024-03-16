use candle_core::{Device, DType, Module};
use hs_mm_embeddings::clip_vit_b_32_image;
fn main() {
    env_logger::init();

    let image_model = clip_vit_b_32_image::model::build_model().unwrap();
    let input_img_batch = candle_core::Tensor::ones((1, 3, 224, 224), DType::F32, &Device::Cpu).unwrap();

    let output = image_model.forward(&input_img_batch).unwrap();
    println!("{:?}", output.dims())
}
