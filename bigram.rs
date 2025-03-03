use tch::{nn, nn::ModuleT, Device, Kind, Tensor};
use tch::nn::OptimizerConfig;
use std::collections::HashMap;
use std::fs;

const BATCH_SIZE: i64 = 32;
const BLOCK_SIZE: i64 = 8;
const MAX_ITERS: i64 = 3000;
const EVAL_INTERVAL: i64 = 300;
const LEARNING_RATE: f64 = 1e-2;
const EVAL_ITERS: i64 = 200;

struct BigramLanguageModel {
    token_embedding_table: nn::Embedding,
}

impl BigramLanguageModel {
    fn new(vs: &nn::Path, vocab_size: i64) -> Self {
        // The embedding maps each token to a vector of size `vocab_size` which directly represents logits.
        let token_embedding_table = nn::embedding(vs, vocab_size, vocab_size, Default::default());
        Self { token_embedding_table }
    }

    // Forward pass returns logits and optionally the cross-entropy loss.
    fn forward(&self, idx: &Tensor, targets: Option<&Tensor>, _train: bool) -> (Tensor, Option<Tensor>) {
        // idx: [B, T]
        let logits = idx.apply(&self.token_embedding_table); // shape: [B, T, vocab_size]
        let loss = if let Some(targets) = targets {
            let (b, t, c) = (logits.size()[0], logits.size()[1], logits.size()[2]);
            let logits_flat = logits.view([b * t, c]);
            let targets_flat = targets.view([b * t]);
            Some(logits_flat.cross_entropy_for_logits(&targets_flat))
        } else {
            None
        };
        (logits, loss)
    }

    // Generate new tokens given an initial context.
    fn generate(&self, idx: &Tensor, max_new_tokens: i64) -> Tensor {
        let mut idx = idx.copy();
        for _ in 0..max_new_tokens {
            let (logits, _) = self.forward(&idx, None, false);
            // Select the logits for the last time step: shape becomes [B, vocab_size]
            let logits = logits.select(1, -1);
            let probs = logits.softmax(-1, Kind::Float);
            let idx_next = probs.multinomial(1, true);
            idx = Tensor::cat(&[idx, idx_next], 1);
        }
        idx
    }
}

// Helper to sample a batch from the data tensor.
fn get_batch(data: &Tensor, device: Device) -> (Tensor, Tensor) {
    let len = data.size()[0];
    let mut xs = Vec::with_capacity(BATCH_SIZE as usize);
    let mut ys = Vec::with_capacity(BATCH_SIZE as usize);
    for _ in 0..BATCH_SIZE {
        let i = i64::from(Tensor::randint(len - BLOCK_SIZE, &[1], (Kind::Int64, device)));
        xs.push(data.narrow(0, i, BLOCK_SIZE));
        ys.push(data.narrow(0, i + 1, BLOCK_SIZE));
    }
    (Tensor::stack(&xs, 0), Tensor::stack(&ys, 0))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use CUDA if available.
    let device = if tch::Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu };

    // Load text from file "input.txt".
    let text = fs::read_to_string("input.txt")?;
    
    // Build the vocabulary from the text.
    let mut chars: Vec<char> = text.chars().collect();
    chars.sort();
    chars.dedup();
    let vocab_size = chars.len() as i64;
    
    // Create mappings between characters and indices.
    let mut stoi = HashMap::new();
    let mut itos = vec![' '; vocab_size as usize];
    for (i, c) in chars.iter().enumerate() {
        stoi.insert(*c, i as i64);
        itos[i] = *c;
    }
    
    // Encode the text as a tensor of indices.
    let data_encoded: Vec<i64> = text.chars().map(|c| stoi[&c]).collect();
    let data = Tensor::of_slice(&data_encoded).to(device);
    
    // Train/validation split.
    let n = (data.size()[0] as f64 * 0.9) as i64;
    let train_data = data.narrow(0, 0, n);
    let val_data = data.narrow(0, n, data.size()[0] - n);

    // Initialize model and optimizer.
    let vs = nn::VarStore::new(device);
    let root = vs.root();
    let model = BigramLanguageModel::new(&root, vocab_size);
    let mut opt = nn::AdamW::default().build(&vs, LEARNING_RATE)?;

    // Training loop.
    for iter in 0..MAX_ITERS {
        // Periodically evaluate the training and validation loss.
        if iter % EVAL_INTERVAL == 0 {
            let mut train_losses = Vec::new();
            for _ in 0..EVAL_ITERS {
                let (xs, ys) = get_batch(&train_data, device);
                let (_, loss_opt) = model.forward(&xs, Some(&ys), false);
                if let Some(loss) = loss_opt {
                    train_losses.push(f64::from(loss));
                }
            }
            let train_loss = train_losses.iter().sum::<f64>() / train_losses.len() as f64;

            let mut val_losses = Vec::new();
            for _ in 0..EVAL_ITERS {
                let (xs, ys) = get_batch(&val_data, device);
                let (_, loss_opt) = model.forward(&xs, Some(&ys), false);
                if let Some(loss) = loss_opt {
                    val_losses.push(f64::from(loss));
                }
            }
            let val_loss = val_losses.iter().sum::<f64>() / val_losses.len() as f64;
            println!("Step {}: train loss {:.4}, val loss {:.4}", iter, train_loss, val_loss);
        }

        // Get a training batch, compute the loss, and update parameters.
        let (xs, ys) = get_batch(&train_data, device);
        let (_, loss_opt) = model.forward(&xs, Some(&ys), true);
        if let Some(loss) = loss_opt {
            opt.zero_grad();
            loss.backward();
            opt.step();
        }
    }

    // Generate text from the trained model.
    let context = Tensor::zeros(&[1, 1], (Kind::Int64, device));
    let generated = model.generate(&context, 500);
    let generated_vec: Vec<i64> = Vec::from(generated);
    let decoded: String = generated_vec.iter().map(|&i| itos[i as usize]).collect();
    println!("{}", decoded);

    Ok(())
}
