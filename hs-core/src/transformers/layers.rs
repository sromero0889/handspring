
// todo!
// Pending following blocks inside TransformerResBlock:

// MultiHeadSelfAttentionLayer (MSP)
// This layer concatenates all the attention outputs linearly to the right dimensions.
// The many attention heads help train local and global dependencies in an image.

// MultiLayerPerceptronsLayer (MLP)
// This layer contains a two-layer with Gaussian Error Linear Unit (GELU)


// LayerNorm -> available in candle




use candle_core::{Module, Tensor};
use candle_nn::{layer_norm, LayerNorm, LayerNormConfig, Linear, linear, VarBuilder};
use crate::errors::HsError;
use crate::errors::HsError::{InitModelError, Other};
use crate::transformers::config::{Activation, MlpLayerConfig, MsaLayerConfig, TransformerLayerConfig, TransformerModelConfig};
use crate::transformers::tensor_ops::TensorOpsExtras;


#[derive(Debug)]
pub struct MlpLayer {
    c_fc: Linear,
    activation: fn(Tensor) -> candle_core::Result<Tensor>,
    c_proj: Linear
}

impl MlpLayer {
    fn new(vb: VarBuilder, config: &MlpLayerConfig) -> Result<Self, HsError> {
        let c_fc = linear(config.hidden_size, config.interm_size, vb.pp(config.c_fc_label)).map_err(InitModelError)?;
        let c_proj = linear(config.interm_size, config.hidden_size, vb.pp(config.c_proj_label)).map_err(InitModelError)?;
        let activation = {
            if let Some(activation) = &config.activation {
                match activation {
                    Activation::QuickGelu => |t: Tensor| {t.quick_gelu()},
                    Activation::Gelu => |t: Tensor| {t.gelu()},
                }
            } else {
                |t: Tensor| {Ok(t)}
            }
        };
        Ok(Self {
            c_fc,
            activation,
            c_proj
        })
    }
}

impl Module for MlpLayer {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // c_fc -> activation (if defined) -> c_proj
        (self.activation)(xs.apply(&self.c_fc)?)?.apply(&self.c_proj)
    }
}

#[derive(Debug)]
pub struct MsaLayer {
    head_dim: usize,
    embedd_dim: usize,
    num_patches: usize,
    num_heads: usize,
    in_proj: Option<Linear>,
    q_k_v_proj: Option<(Linear, Linear, Linear)>,
    out_proj: Linear
}

impl MsaLayer {

    fn apply_in_proj(&self, xs: &Tensor) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        // xs: [sequence_length, b_size, embedd_dim]
        if let Some(in_proj) = &self.in_proj {
            // in_proj.w: [3 * embedd_dim, embedd_dim]
            // in_proj.b: [3 * embedd_dim]
            let xs = xs.apply(in_proj)?;
            // xs: [sequence_length, b_size, 3 * embedd_dim]
            let (seq_len,batch_size,embedd_dim_3x)= xs.dims3()?;
            let embedd_dim = embedd_dim_3x/3;

            let q_k_v = xs.reshape((seq_len, batch_size, 3, embedd_dim))?
                // [seq_len, batch_size, 3, embedd_dim] same as unflatten last dim
                .unsqueeze(0)?
                // [1, seq_len, batch_size, 3, embedd_dim]
                .transpose(0, 3)?
                // [3, seq_len, batch_size, 1, embedd_dim]
                .squeeze(3)?
                // [3, seq_len, batch_size, embedd_dim]
                .contiguous()? // ??
                .chunk(3, 0)?;
                // Vector of 3 tensors of [1, seq_len, batch_size, embedd_dim]



            match q_k_v.as_slice() {
                [q, k, v] => {
                    // reshape q, k, v for multihead attention and make them batch first
                    // num_heads * head_dim = embedd_dim -> unflatten last dim
                    let q = q.reshape((seq_len, batch_size, self.num_heads, self.head_dim))?
                        // [seq_len, batch_size, num_heads, head_dim]
                        .permute((1,2,0,3))?;
                        // [batch_size, num_heads, seq_len, head_dim]
                    let k = k.reshape((seq_len, batch_size, self.num_heads, self.head_dim))?
                        // [seq_len, batch_size, num_heads, head_dim]
                        .permute((1,2,0,3))?;
                        // [batch_size, num_heads, seq_len, head_dim]
                    let v = v.reshape((seq_len, batch_size, self.num_heads, self.head_dim))?
                        // [seq_len, batch_size, num_heads, head_dim]
                        .permute((1,2,0,3))?;
                        // [batch_size, num_heads, seq_len, head_dim]

                    Ok((q, k, v))
                },
                _ => {
                    Err(candle_core::error::Error::Msg(String::from("Error MsaLayer: q_k_v projections vector incorrect")))
                }
            }
        } else if let Some((q_proj, k_proj, v_proj)) = &self.q_k_v_proj {
            let q = xs.apply(q_proj)?;
            let k = xs.apply(k_proj)?;
            let v = xs.apply(v_proj)?;
            // todo!()
            Ok((q, k, v))
        } else {

            Err(candle_core::error::Error::Msg(String::from("Error MsaLayer: neither in_proj or q_k_v_proj has value")))
        }
    }
    fn new(vb: VarBuilder, config: &MsaLayerConfig) -> Result<Self, HsError> {

        let (in_proj, q_k_v_proj) = if let Some(in_proj_key) = config.in_proj_label {
            let in_proj = linear(
                config.embed_dim,
                config.interm_size,
                vb.pp(in_proj_key)
            ).map_err(InitModelError)?;

            (Some(in_proj), None)

        } else if let (Some(in_proj_w_label) , Some (in_proj_b_label)) = (config.in_proj_w_label, config.in_proj_b_label) {
            // todo, this path will be removed when corrected labels in safetensors file
            let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
            let ws = vb.get_with_hints((config.interm_size, config.embed_dim), in_proj_w_label, init_ws).map_err(InitModelError)?;
            let bound = 1. / (config.embed_dim as f64).sqrt();
            let init_bs = candle_nn::Init::Uniform {
                lo: -bound,
                up: bound,
            };
            let bs = vb.get_with_hints(config.interm_size, in_proj_b_label, init_bs).map_err(InitModelError)?;
            let in_proj = Linear::new(ws, Some(bs));
            (Some(in_proj), None)

        } else if let (Some(q_label) , Some (k_label), Some(v_label)) = (config.q_label, config.k_label, config.v_label) {
            let embed_dim = config.embed_dim / 3;
            let interm_size = config.interm_size / 3; // Todo check
            (None, Some((
                linear(
                    embed_dim,
                    interm_size,
                    vb.pp(q_label)
                ).map_err(InitModelError)?,
                linear(
                    embed_dim,
                    interm_size,
                    vb.pp(k_label)
                ).map_err(InitModelError)?,
                linear(
                    embed_dim,
                    interm_size,
                    vb.pp(v_label)
                ).map_err(InitModelError)?
            )))
        } else {
            (None, None)
        };


        let out_proj= linear(
            config.embed_dim,
            config.embed_dim,
            vb.pp(config.out_proj_label)
        ).map_err(InitModelError)?;

        if in_proj.is_none() && q_k_v_proj.is_none() {
            Err(Other("MsaLayer Error: Could not initialize model, wrong q,k,v proj parameters"))
        } else{
            Ok(Self {
                // Todo
                head_dim: config.head_dim,
                embedd_dim: config.embed_dim,
                num_patches: config.num_patches,
                num_heads: config.num_heads,
                in_proj,
                q_k_v_proj,
                out_proj
            })
        }
    }
}

impl Module for MsaLayer {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (q, k, v) = self.apply_in_proj(xs)?;
        let head_dim = self.head_dim as f64; // num_heads = vision_width // 64 -> head_dim = embed_dim // num_heads
        // q,k,v: [batch_size, num_heads, seq_len, head_dim]
        let (batch_size, _, seq_len, _) = q.dims4()?;
        // Todo add attention_mask for text
        let xs = Tensor::scaled_dot_product_attn(&q, &k, &v, head_dim)?;
        // xs: [batch_size, num_heads, seq_len, head_dim]
        let xs = xs.permute((2,0,1,3))?
            // [seq_len, batch_size, num_heads, head_dim]
            .reshape((seq_len * batch_size, self.embedd_dim))?;
            // [seq_len * batch_size, num_heads * head_dim = embedd_dim]

        xs.apply(&self.out_proj)?
            // [seq_len,  batch_size, embedd_dim]
            .reshape((seq_len, batch_size, self.embedd_dim))
    }
}

pub trait TransformerEncoderLayer {
    fn new(vb: VarBuilder, config: &TransformerLayerConfig) -> Result<Self, HsError> where Self: Sized;
}

#[derive(Debug)]
pub struct TransformerResBlock {
    ln_1: LayerNorm,
    msa_layer: MsaLayer,
    ln_2: LayerNorm,
    mlp_layer: MlpLayer,

}

impl TransformerEncoderLayer for TransformerResBlock {
    fn new(vb: VarBuilder, config: &TransformerLayerConfig) -> Result<Self, HsError> where Self: Sized {
        let ln_1 = layer_norm(
            config.hidden_size,
            LayerNormConfig::default(),
            vb.pp(config.ln_1_label)
        ).map_err(InitModelError)?;
        let ln_2 = layer_norm(
            config.hidden_size,
            LayerNormConfig::default(),
            vb.pp(config.ln_2_label)
        ).map_err(InitModelError)?;

        let mlp_layer = MlpLayer::new(vb.pp(config.mlp_label), &config.mlp_layer)?;
        let msa_layer = MsaLayer::new(vb.pp(config.msa_label), &config.msa_layer)?;

        Ok(Self {
            ln_1,
            msa_layer,
            ln_2,
            mlp_layer,
        })
    }
}

impl Module for TransformerResBlock {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let b = xs.apply(&self.ln_1)?.apply(&self.msa_layer)?;
        // b [seq_len,  batch_size, embedd_dim]
        let xs = xs.add(&b)?;
        let b = xs.apply(&self.ln_2)?.apply(&self.mlp_layer)?;
        xs.add(&b)
        // [seq_len,  batch_size, embedd_dim]
    }
}

pub trait GenEmbeddLayer {
    fn new<C: TransformerModelConfig>(vb: VarBuilder, config: &C) -> Result<Self, HsError> where Self: Sized;
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use candle_core::{Device, DType, Tensor};
    use candle_nn::{VarBuilder};
    use crate::transformers::config::MlpLayerConfig;
    use crate::transformers::config::Activation::QuickGelu;

    use super::*;

    #[test]
    fn mlp_layer_ok() {
        let config = MlpLayerConfig {
            hidden_size: 768,
            interm_size: 3072,
            activation: Some(QuickGelu),
            c_fc_label: "c_fc",
            c_proj_label: "c_proj",
        };
        let mut ts: HashMap<String, Tensor> = HashMap::new();
        ts.insert(String::from("c_fc.weight"), Tensor::ones((3072, 768), DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("c_fc.bias"), Tensor::ones(3072, DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("c_proj.weight"), Tensor::ones((768, 3072), DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("c_proj.bias"), Tensor::ones(768, DType::F32, &Device::Cpu).unwrap());

        let vb = VarBuilder::from_tensors(ts, DType::F32, &Device::Cpu);

        let layer = MlpLayer::new(vb, &config).unwrap();

        let input = Tensor::ones((50,2,768), DType::F32, &Device::Cpu).unwrap();
        let res = input.apply(&layer).unwrap();
        assert_eq!(res.dims3().unwrap(), (50,2,768));
    }

    #[test]
    fn msa_layer_ok() {
        let config = MsaLayerConfig {
            embed_dim: 768,
            head_dim: 64,
            num_patches: 50,
            num_heads: 12,
            interm_size: 2304,
            in_proj_label: None,
            in_proj_w_label: Some("in_proj_weight"),
            in_proj_b_label: Some("in_proj_bias"),
            q_label: None,
            k_label: None,
            v_label: None,
            out_proj_label: "out_proj",
        };
        let mut ts: HashMap<String, Tensor> = HashMap::new();
        ts.insert(String::from("in_proj_weight"), Tensor::ones((2304, 768), DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("in_proj_bias"), Tensor::ones(2304, DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("out_proj.weight"), Tensor::ones((768, 768), DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("out_proj.bias"), Tensor::ones(768, DType::F32, &Device::Cpu).unwrap());

        let vb = VarBuilder::from_tensors(ts, DType::F32, &Device::Cpu);

        let layer = MsaLayer::new(vb, &config).unwrap();
        let input = Tensor::ones((50,2,768), DType::F32, &Device::Cpu).unwrap();
        let res = input.apply(&layer).unwrap();
        assert_eq!(res.dims3().unwrap(), (50,2,768));
    }

    #[test]
    fn transformer_res_block_ok(){
        // TransformerResBlock
        todo!()
    }
}