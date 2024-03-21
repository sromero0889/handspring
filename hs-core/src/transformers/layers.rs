
// MultiHeadSelfAttentionLayer (MSP)
// This layer concatenates all the attention outputs linearly to the right dimensions.
// The many attention heads help train local and global dependencies in an image.

// MultiLayerPerceptronsLayer (MLP)
// This layer contains a two-layer with Gaussian Error Linear Unit (GELU)


// LayerNorm -> available in candle




use std::collections::HashMap;
use candle_core::{D, Device, DType, Error, IndexOp, Module, Tensor};
use candle_nn::{layer_norm, LayerNorm, LayerNormConfig, Linear, linear, VarBuilder};
use crate::errors::HsError;
use crate::errors::HsError::InitModelError;
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
        let c_fc = linear(config.hidden_size, config.interm_size, vb.pp(config.c_fc_label.as_str())).map_err(InitModelError)?;
        let c_proj = linear(config.interm_size, config.hidden_size, vb.pp(config.c_proj_label.as_str())).map_err(InitModelError)?;
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
    num_heads: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    in_proj: Linear,
    out_proj: Linear
}

impl MsaLayer {

    // fn apply_in_proj(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
    //     // Todo!!
    //     // q, l, v: [sequence_length, b_size, embedd_dim]
    //     // let (seq_len,batch_size, _)= xs.dims3()?;
    //     // let q = q.apply(&self.q_proj)?;
    //     // let k = k.apply(&self.k_proj)?;
    //     // let v = v.apply(&self.v_proj)?;
    //
    //     unimplemented!()
    // }

    fn apply_in_proj_packed(&self, xs: &Tensor) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        // xs: [sequence_length, b_size, embedd_dim]
        let (seq_len,batch_size, embedd_dim)= xs.dims3()?;
        let proj = xs.apply(&self.in_proj)?
            .reshape((seq_len, batch_size, 3, embedd_dim))?
            .unsqueeze(0)?
            .transpose(0, D::Minus2)?
            .squeeze(D::Minus2)?.
            contiguous()?;

        let q = proj.i(0)?;
        let k = proj.i(1)?;
        let v = proj.i(2)?;
        // q = k = v: [sequence_length, b_size, embedd_dim]
        Ok((q, k, v))
    }
    fn reshape_before_mask(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // q = k = v: [sequence_length, b_size, embedd_dim]
        let (seq_len, b_size, _) = xs.dims3()?;
        xs.reshape((seq_len, b_size * self.num_heads, self.head_dim))?.transpose(0,1)
        // [b_size * num_heads, sequence_length, head_dim]
    }

    fn reshape_after_mask(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // [b_size * num_heads, sequence_length, head_dim]
        let (dim0, seq_len, _) = xs.dims3()?;
        let b_size = dim0 / self.num_heads;
        xs.reshape((b_size, self.num_heads, seq_len, self.head_dim))
        // [b_size, num_heads, sequence_length, head_dim]
    }

    fn process_qkv(&self, t: & Tensor) -> candle_core::Result<Tensor> {
        let t = self.reshape_before_mask(&t)?;
        self.reshape_after_mask(&t)
    }



    fn new(vb: VarBuilder, config: &MsaLayerConfig) -> Result<Self, HsError> {

        let (vb_in_proj, vb_q_proj, vb_k_proj, vb_v_proj) = if let Some(in_proj_key) = &config.in_proj_label {


            // In this case interm_size = 3 * embedd_dim
            let w = vb.get((config.interm_size,config.embed_dim), format!("{}.weight", in_proj_key).as_str()).unwrap();
            let b = vb.get(config.interm_size, format!("{}.bias", in_proj_key).as_str()).map_err(InitModelError)?;

            let mut vb_proj_layers: Vec<VarBuilder> = Vec::new();

            for offset in 0..3 {
                let w_proj = w.i(offset*config.embed_dim..(offset + 1)*config.embed_dim).map_err(InitModelError)?;
                let b_proj = b.i(offset*config.embed_dim..(offset + 1)*config.embed_dim).map_err(InitModelError)?;
                let mut proj_ts: HashMap<String, Tensor> = HashMap::new();
                proj_ts.insert(String::from("weight"), w_proj);
                proj_ts.insert(String::from("bias"), b_proj);
                vb_proj_layers.push(VarBuilder::from_tensors(proj_ts, DType::F32, &Device::Cpu));

            };

            match &vb_proj_layers[..] {
                [q_proj, k_proj, v_proj] => Ok((vb.pp(in_proj_key), q_proj.to_owned(), k_proj.to_owned(), v_proj.to_owned())),
                _ => Err(Error::Msg(String::from("Problem building projection layers in MSA block")))
            }


        } else if let (Some(q_label) , Some (k_label), Some(v_label)) = (&config.q_label, &config.k_label, &config.v_label) {
            let wq_proj = vb.get((config.embed_dim, config.embed_dim), format!("{}.weight", q_label).as_str()).map_err(InitModelError)?;
            let bq_proj = vb.get(config.embed_dim, format!("{}.weight", q_label).as_str()).map_err(InitModelError)?;
            let wk_proj = vb.get((config.embed_dim, config.embed_dim), format!("{}.weight", k_label).as_str()).map_err(InitModelError)?;
            let bk_proj = vb.get(config.embed_dim, format!("{}.weight", k_label).as_str()).map_err(InitModelError)?;
            let wv_proj = vb.get((config.embed_dim, config.embed_dim), format!("{}.weight", v_label).as_str()).map_err(InitModelError)?;
            let bv_proj = vb.get(config.embed_dim, format!("{}.weight", v_label).as_str()).map_err(InitModelError)?;

            let w_in_proj = Tensor::cat(&[wq_proj, wk_proj, wv_proj], 0).map_err(InitModelError)?;
            let v_in_proj = Tensor::cat(&[bq_proj, bk_proj, bv_proj], 0).map_err(InitModelError)?;
            let mut in_proj_ts: HashMap<String, Tensor> = HashMap::new();
            in_proj_ts.insert(String::from("weight"), w_in_proj);
            in_proj_ts.insert(String::from("bias"), v_in_proj);
            let vb_in_proj = VarBuilder::from_tensors(in_proj_ts, DType::F32, &Device::Cpu);
            Ok(
                (vb_in_proj,
                 vb.pp(q_label),
                 vb.pp(k_label),
                 vb.pp(v_label)
                ))
        } else {
            Err(Error::Msg(String::from("Problem building projection layers in MSA block, check config.json & model.safetensors")))
        }.map_err(InitModelError)?;


        let out_proj= linear(
            config.embed_dim,
            config.embed_dim,
            vb.pp(config.out_proj_label.as_str())
        ).map_err(InitModelError)?;

        Ok(Self {
            // Todo
            head_dim: config.head_dim,
            embedd_dim: config.embed_dim,
            // num_patches: config.num_patches,
            num_heads: config.num_heads,
            q_proj: linear(
                config.embed_dim,
                config.embed_dim,
                vb_q_proj
            ).map_err(InitModelError)?,
            k_proj: linear(
                config.embed_dim,
                config.embed_dim,
                vb_k_proj
            ).map_err(InitModelError)?,
            out_proj,
            v_proj: linear(
                config.embed_dim,
                config.embed_dim,
                vb_v_proj
            ).map_err(InitModelError)?,
            in_proj: linear(
                config.embed_dim,
                3 * config.embed_dim,
                vb_in_proj
            ).map_err(InitModelError)?,
        })
    }
}
use rayon_join_macro::join;
impl Module for MsaLayer {


    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // todo conditional
        let (q, k, v) = self.apply_in_proj_packed(xs)?;
        // let q = self.reshape_before_mask(&q)?;
        // let k = self.reshape_before_mask(&k)?;
        // let v = self.reshape_before_mask(&v)?;
        // [b_size * num_heads, sequence_length, head_dim]
        // Todo add attention_mask for text

        // let q = self.reshape_after_mask(&q?)?;
        // let k = self.reshape_after_mask(&k?)?;
        // let v = self.reshape_after_mask(&v?)?;
        // [b_size, num_heads, sequence_length, head_dim]



        let (q, k, v) = join!(
            || self.process_qkv(&q),
            || self.process_qkv(&k),
            || self.process_qkv(&v)
        );

        let head_dim = self.head_dim as f64; // num_heads = vision_width // 64 -> head_dim = embed_dim // num_heads

        let xs = Tensor::scaled_dot_product_attn(&q?, &k?, &v?, head_dim)?;
        // xs: [batch_size, num_heads, sequence_length, head_dim]

        let (b_size, _, seq_len, _) = xs.dims4()?;
        let xs = xs.permute((2,0,1,3))?
            // [seq_len, batch_size, num_heads, head_dim]
            .reshape((b_size * seq_len, self.embedd_dim))?;
            // [seq_len * batch_size, num_heads * head_dim = embedd_dim]

        xs.apply(&self.out_proj)?
            // [seq_len,  batch_size, embedd_dim]
            .reshape((seq_len, b_size, self.embedd_dim))
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
            vb.pp(config.ln_1_label.as_str())
        ).map_err(InitModelError)?;
        let ln_2 = layer_norm(
            config.hidden_size,
            LayerNormConfig::default(),
            vb.pp(config.ln_2_label.as_str())
        ).map_err(InitModelError)?;

        let mlp_layer = MlpLayer::new(vb.pp(config.mlp_label.as_str()), &config.mlp_layer)?;
        let msa_layer = MsaLayer::new(vb.pp(config.msa_label.as_str()), &config.msa_layer)?;

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
            c_fc_label: String::from("c_fc"),
            c_proj_label: String::from("c_proj"),
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
            in_proj_label: Some(String::from("in_proj")),
            q_label: None,
            k_label: None,
            v_label: None,
            out_proj_label: String::from("out_proj"),
        };
        let mut ts: HashMap<String, Tensor> = HashMap::new();
        ts.insert(String::from("in_proj.weight"), Tensor::ones((2304, 768), DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("in_proj.bias"), Tensor::ones(2304, DType::F32, &Device::Cpu).unwrap());
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
        let config = TransformerLayerConfig {
            hidden_size: 768,
            ln_1_label: String::from("ln_1"),
            ln_2_label: String::from("ln_2"),
            msa_label: String::from("attn"),
            mlp_label: String::from("mlp"),
            mlp_layer: MlpLayerConfig {
                hidden_size: 768,
                interm_size: 3072,
                activation: Some(QuickGelu),
                c_fc_label: String::from("c_fc"),
                c_proj_label: String::from("c_proj"),
            },
            msa_layer: MsaLayerConfig {
                embed_dim: 768,
                head_dim: 64,
                num_patches: 49,
                num_heads: 12,
                interm_size: 2304,
                in_proj_label: Some(String::from("in_proj")),
                q_label: None,
                k_label: None,
                v_label: None,
                out_proj_label: String::from("out_proj"),
            },
        };

        let mut ts: HashMap<String, Tensor> = HashMap::new();

        ts.insert(String::from("ln_1.weight"), Tensor::ones(768, DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("ln_1.bias"), Tensor::ones(768, DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("ln_2.weight"), Tensor::ones( 768, DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("ln_2.bias"), Tensor::ones(768, DType::F32, &Device::Cpu).unwrap());

        ts.insert(String::from("mlp.c_fc.weight"), Tensor::ones((3072, 768), DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("mlp.c_fc.bias"), Tensor::ones(3072, DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("mlp.c_proj.weight"), Tensor::ones((768, 3072), DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("mlp.c_proj.bias"), Tensor::ones(768, DType::F32, &Device::Cpu).unwrap());

        ts.insert(String::from("attn.in_proj.weight"), Tensor::ones((2304, 768), DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("attn.in_proj.bias"), Tensor::ones(2304, DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("attn.out_proj.weight"), Tensor::ones((768, 768), DType::F32, &Device::Cpu).unwrap());
        ts.insert(String::from("attn.out_proj.bias"), Tensor::ones(768, DType::F32, &Device::Cpu).unwrap());

        let vb = VarBuilder::from_tensors(ts, DType::F32, &Device::Cpu);

        let resblock = TransformerResBlock::new(vb, &config).unwrap();
        let input = Tensor::ones((50,2,768), DType::F32, &Device::Cpu).unwrap();
        let res = input.apply(&resblock).unwrap();
        assert_eq!(res.dims3().unwrap(), (50,2,768));

    }


}