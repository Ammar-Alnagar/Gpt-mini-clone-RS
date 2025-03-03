
```markdown
# Bigram Language Model in Rust

This project implements a simple character-level bigram language model using Rust and the [tch crate](https://github.com/LaurentMazare/tch-rs), which provides Rust bindings for PyTorch. The model is trained on the Tiny Shakespeare dataset, predicting the next character based solely on the current character.

## Features

- **Bigram Language Model:** Uses a single embedding layer to map each token directly to logits over the vocabulary.
- **Training & Evaluation:** Implements a training loop with periodic evaluation on training and validation sets.
- **Text Generation:** Generates new text sequences by sampling from the model’s predictions.
- **Rust & tch:** Leverages Rust’s performance and type safety along with the flexibility of PyTorch via the tch crate.

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (latest stable version recommended)
- [Cargo](https://doc.rust-lang.org/cargo/)
- Optionally, CUDA if you want to run on GPU (the code will fall back to CPU if CUDA is unavailable)

## Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/bigram-language-model.git
   cd bigram-language-model
   ```

2. **Download the Dataset:**

   Download the Tiny Shakespeare dataset and place the file (`input.txt`) in the root directory of the project:

   ```bash
   wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
   ```

3. **Build the Project:**

   Build the project using Cargo:

   ```bash
   cargo build --release
   ```

## Running the Model

Run the project with Cargo. This will start the training process and, once completed, generate sample text:

```bash
cargo run --release
```

During training, the program will periodically print out training and validation losses. After training, it will output generated text to the console.

## Project Structure

- **`main.rs`**: Contains the implementation of the Bigram Language Model, including data preprocessing, training loop, and text generation.
- **`Cargo.toml`**: Defines the project dependencies (including the `tch` crate).

## Hyperparameters

The following hyperparameters are defined in the code:

- **Batch size:** 32
- **Block size:** 8
- **Max iterations:** 3000
- **Evaluation interval:** 300 iterations
- **Learning rate:** 1e-2
- **Evaluation iterations:** 200

Feel free to modify these settings in the source code to experiment with different configurations.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [tch crate](https://github.com/LaurentMazare/tch-rs)
- [Tiny Shakespeare dataset](https://github.com/karpathy/char-rnn)
```
