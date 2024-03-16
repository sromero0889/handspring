#[macro_use] extern crate criterion;
 // https://bheisler.github.io/criterion.rs/book/criterion_rs.html
 // todo: These are only examples of benches, design proper Metrics to evaluate performance
use criterion::*;

fn encode(c: &mut Criterion) {
    use candle_core::{Device, DType, Module};
    use hs_mm_embeddings::clip_vit_b_32_image;

    let image_model = clip_vit_b_32_image::model::build_model().unwrap();
    let input_img_batch = candle_core::Tensor::ones((1, 3, 224, 224), DType::F32, &Device::Cpu).unwrap();


    c.bench_function("image_model.forward", move |b| {
        b.iter(|| image_model.forward(&input_img_batch).unwrap())
    });
}

fn init_model(c: &mut Criterion) {
    use hs_mm_embeddings::clip_vit_b_32_image;

    c.bench_function(" clip_vit_b_32_image::model::build_model()", move |b| {
        b.iter(|| clip_vit_b_32_image::model::build_model().unwrap())
    });
}


criterion_group!{
    name = benches;
    // This can be any expression that returns a `Criterion` object.
    config = Criterion::default().sample_size(10);
    targets = init_model, encode
}
criterion_main!(benches);
