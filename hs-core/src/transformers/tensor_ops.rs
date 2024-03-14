use candle_core::Tensor;

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
        let q = q.contiguous()?;
        let kt = k.t()?;
        let kt= if kt.is_contiguous() { kt } else { kt.contiguous()? };
        let v = v.contiguous()?;

        candle_nn::ops::softmax_last_dim(&(q.matmul(&kt)? / f64::sqrt(dk))?)?
            .matmul(&v)

    }
}