use candle_core::{D, Tensor};

pub trait TensorOpsExtras {
    fn quick_gelu(&self) -> candle_core::Result<Tensor>;
    fn scaled_dot_product_attn(q: &Tensor, k: &Tensor, v: &Tensor, dk: f64) -> candle_core::Result<Tensor>;
}

impl TensorOpsExtras for Tensor {
    fn quick_gelu(&self) -> candle_core::Result<Tensor> {
        // from Python -> x * torch.sigmoid(1.702 * x)
        self * candle_nn::ops::sigmoid(&(self * 1.702f64)?)?
    }

    //https://arxiv.org/pdf/1706.03762.pdf
    fn scaled_dot_product_attn(q: &Tensor, k: &Tensor, v: &Tensor, dk: f64) -> candle_core::Result<Tensor> {

        // matmul is an operation that requires tensors to be contiguous
        // let q = q.contiguous()?;
        let kt = k.transpose(D::Minus2, D::Minus1)?;
        // let kt= if kt.is_contiguous() { kt } else { kt.contiguous()? };
        // let v = v.contiguous()?;
        let scale_factor = 1. / f64::sqrt(dk);

        candle_nn::ops::softmax_last_dim(&(q.matmul(&kt)? * scale_factor)?)?
            .matmul(&v)
    }
}