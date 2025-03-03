use tch::{nn, nn::ModuleT, Device, Kind, Tensor};
use tch::nn::OptimizerConfig;
use std::collections::HashMap;
use std::fs;

const BATCH_SIZE: i64 = 64;
const BLOCK_SIZE: i64 = 256;
const MAX_ITERS: i64 = 5000;
const EVAL_INTERVAL: i64 = 500;
const LEARNING_RATE: f64 = 5e-5;
const N_EMBD: i64 = 384;
const N_HEAD: i64 = 6;
const N_LAYER: i64 = 6;
const DROPOUT: f64 = 0.1;

//
// Self-Attention Head
//
struct Head {
    key: nn::Linear,
    query: nn::Linear,
    value: nn::Linear,
    tril: Tensor,
    dropout: f64,
}

impl Head {
    fn new(vs: &nn::Path, head_size: i64) -> Self {
        let key = nn::linear(vs, N_EMBD, head_size, nn::LinearConfig { bias: false });
        let query = nn::linear(vs, N_EMBD, head_size, nn::LinearConfig { bias: false });
        let value = nn::linear(vs, N_EMBD, head_size, nn::LinearConfig { bias: false });
        let tril = Tensor::ones(&[BLOCK_SIZE, BLOCK_SIZE], (Kind::Float, vs.device())).tril(0);
        Self { key, query, value, tril, dropout: DROPOUT }
    }

    fn forward(&self, xs: &Tensor, train: bool) -> Tensor {
        // xs shape: [batch, time, channels]
        let k = xs.apply(&self.key);   // [B, T, head_size]
        let q = xs.apply(&self.query);   // [B, T, head_size]
        // compute attention scores (scaled dot-product)
        let scale = (k.size()[2] as f64).sqrt();
        let mut wei = q.matmul(&k.transpose(-2, -1)) / scale;
        let t = xs.size()[1];
        // mask out future positions (only use lower triangular part)
        let mask = self.tril.i((..t, ..t));
        wei = wei.masked_fill(&mask.eq(0), f64::NEG_INFINITY);
        wei = wei.softmax(-1, Kind::Float);
        if train {
            wei = wei.dropout(self.dropout, train);
        }
        // weighted aggregation of the values
        let v = xs.apply(&self.value);
        wei.matmul(&v)
    }
}

//
// Multi-Head Attention
//
struct MultiHeadAttention {
    heads: Vec<Head>,
    proj: nn::Linear,
    dropout: f64,
}

impl MultiHeadAttention {
    fn new(vs: &nn::Path, num_heads: i64, head_size: i64) -> Self {
        let mut heads = Vec::new();
        for i in 0..num_heads {
            heads.push(Head::new(&vs / format!("head_{}", i), head_size));
        }
        let proj = nn::linear(vs, num_heads * head_size, N_EMBD, Default::default());
        Self { heads, proj, dropout: DROPOUT }
    }

    fn forward(&self, xs: &Tensor, train: bool) -> Tensor {
        let head_outs: Vec<Tensor> = self.heads.iter().map(|h| h.forward(xs, train)).collect();
        let cat = Tensor::cat(&head_outs, -1);
        let mut out = cat.apply(&self.proj);
        if train {
            out = out.dropout(self.dropout, train);
        }
        out
    }
}

//
// Feed-Forward Network
//
struct FeedForward {
    net: nn::Sequential,
}

impl FeedForward {
    fn new(vs: &nn::Path) -> Self {
        let net = nn::seq()
            .add(nn::linear(vs, N_EMBD, 4 * N_EMBD, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(vs, 4 * N_EMBD, N_EMBD, Default::default()))
            .add_fn(|x| x.dropout(DROPOUT, true));
        Self { net }
    }

    fn forward(&self, xs: &Tensor, train: bool) -> Tensor {
        self.net.forward_t(xs, train)
    }
}

//
// Transformer Block
//
struct Block {
    ln1: nn::LayerNorm,
    ln2: nn::LayerNorm,
    sa: MultiHeadAttention,
    ffwd: FeedForward,
}

impl Block {
    fn new(vs: &nn::Path) -> Self {
        let ln_config = nn::LayerNormConfig { eps: 1e-5, ..Default::default() };
        let ln1 = nn::layer_norm(vs, vec![N_EMBD], ln_config);
        let ln2 = nn::layer_norm(vs, vec![N_EMBD], ln_config);
        let head_size = N_EMBD / N_HEAD;
        let sa = MultiHeadAttention::new(&vs / "sa", N_HEAD, head_size);
        let ffwd = FeedForward::new(&vs / "ffwd");
        Self { ln1, ln2, sa, ffwd }
    }

    fn forward(&self, xs: &Tensor, train: bool) -> Tensor {
        let a = self.sa.forward(&xs.apply(&self.ln1), train);
        let xs = xs + a;
        let b = self.ffwd.forward(&xs.apply(&self.ln2), train);
        xs + b
    }
}

//
// GPT Language Model
//
struct GPTLanguageModel {
    token_embedding_table: nn::Embedding,
    position_embedding_table: nn::Embedding,
    blocks: Vec<Block>,
    ln_f: nn::LayerNorm,
    lm_head: nn::Linear,
}

impl GPTLanguageModel {
    fn new(vs: &nn::Path, vocab_size: i64) -> Self {
        let token_embedding_table = nn::embedding(vs, vocab_size, N_EMBD, Default::default());
        let position_embedding_table = nn::embedding(vs, BLOCK_SIZE, N_EMBD, Default::default());
        let mut blocks = Vec::new();
        for i in 0..N_LAYER {
            blocks.push(Block::new(&vs / format!("block_{}", i)));
        }
        let ln_f = nn::layer_norm(vs, vec![N_EMBD], Default::default());
        let lm_head = nn::linear(vs, N_EMBD, vocab_size, Default::default());
        Self { token_embedding_table, position_embedding_table, blocks, ln_f, lm_head }
    }

    fn forward(&self, idx: &Tensor, targets: Option<&Tensor>, train: bool) -> (Tensor, Option<Tensor>) {
        // idx shape: [B, T]
        let (b, t) = (idx.size()[0], idx.size()[1]);
        let tok_emb = idx.apply(&self.token_embedding_table);
        // positions: [0, 1, ..., t-1]
        let pos = Tensor::arange(t, (Kind::Int64, idx.device()));
        let pos_emb = pos.apply(&self.position_embedding_table);
        let mut x = tok_emb + pos_emb;
        for block in &self.blocks {
            x = block.forward(&x, train);
        }
        x = x.apply(&self.ln_f);
        let logits = x.apply(&self.lm_head);
        let loss = targets.map(|tgt| {
            let logits_flat = logits.view([b * t, -1]);
            let targets_flat = tgt.view([b * t]);
            logits_flat.cross_entropy_for_logits(&targets_flat)
        });
        (logits, loss)
    }

    // Generate new tokens given an initial context.
    fn generate(&self, idx: &Tensor, max_new_tokens: i64) -> Tensor {
        let mut idx = idx.copy();
        for _ in 0..max_new_tokens {
            let cur_t = idx.size()[1];
            let idx_cond = if cur_t > BLOCK_SIZE {
                idx.narrow(1, cur_t - BLOCK_SIZE, BLOCK_SIZE)
            } else {
                idx.shallow_clone()
            };
            let (logits, _) = self.forward(&idx_cond, None, false);
            // select the logits for the last time step
            let logits = logits.select(1, -1);
            let probs = logits.softmax(-1, Kind::Float);
            let idx_next = probs.multinomial(1, true);
            idx = Tensor::cat(&[idx, idx_next], 1);
        }
        idx
    }
}

//
// Helper: create a batch from data
//
fn get_batch(data: &Tensor, device: Device) -> (Tensor, Tensor) {
    let len = data.size()[0];
    let mut xs_vec = Vec::with_capacity(BATCH_SIZE as usize);
    let mut ys_vec = Vec::with_capacity(BATCH_SIZE as usize);
    for _ in 0..BATCH_SIZE {
        let i = Tensor::randint(len - BLOCK_SIZE, &[], (Kind::Int64, device));
        let start = i.int64_value(&[]);
        xs_vec.push(data.i(start..start + BLOCK_SIZE));
        ys_vec.push(data.i(start + 1..start + BLOCK_SIZE + 1));
    }
    (Tensor::stack(&xs_vec, 0), Tensor::stack(&ys_vec, 0))
}

//
// Main training and generation loop
//
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = if tch::Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu };

    // Load text file (make sure "input.txt" is in the working directory)
    let text = fs::read_to_string("input.txt")?;
    // Build vocabulary from the text.
    let mut chars: Vec<char> = text.chars().collect();
    chars.sort_unstable();
    chars.dedup();
    let vocab_size = chars.len() as i64;
    let stoi: HashMap<char, i64> = chars.iter().enumerate().map(|(i, &c)| (c, i as i64)).collect();
    let itos: Vec<char> = {
        let mut v = vec![' '; vocab_size as usize];
        for (&c, &i) in &stoi {
            v[i as usize] = c;
        }
        v
    };
    // Encoder and decoder closures.
    let encode = |s: &str| -> Vec<i64> { s.chars().map(|c| stoi[&c]).collect() };
    let decode = |indices: &Tensor| -> String {
        let data: Vec<i64> = Vec::<i64>::from(indices);
        data.into_iter().map(|i| itos[i as usize]).collect()
    };

    // Encode full text as tensor and split into train/val.
    let data_encoded = encode(&text);
    let data_tensor = Tensor::of_slice(&data_encoded).to(device);
    let n = (data_tensor.size()[0] as f64 * 0.9) as i64;
    let train_data = data_tensor.i(..n);
    let _val_data = data_tensor.i(n..);

    // Build the model.
    let vs = nn::VarStore::new(device);
    let root = vs.root();
    let model = GPTLanguageModel::new(&root, vocab_size);

    // Create the optimizer.
    let mut opt = nn::AdamW::default().build(&vs, LEARNING_RATE)?;

    // Training loop.
    for iter in 0..MAX_ITERS {
        if iter % EVAL_INTERVAL == 0 {
            // Evaluate on a batch from training data.
            let (xs, ys) = get_batch(&train_data, device);
            let (_logits, loss_opt) = model.forward(&xs, Some(&ys), false);
            if let Some(loss) = loss_opt {
                println!("Step {}: train loss {:.4}", iter, f64::from(loss));
            }
        }
        let (xs, ys) = get_batch(&train_data, device);
        let (_logits, loss_opt) = model.forward(&xs, Some(&ys), true);
        if let Some(loss) = loss_opt {
            opt.zero_grad();
            loss.backward();
            opt.step();
        }
    }

    // Generate text from the model.
    let context = Tensor::zeros(&[1, 1], (Kind::Int64, device));
    let generated = model.generate(&context, 500);
    println!("{}", decode(&generated));

    Ok(())
}
