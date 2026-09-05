# PII Detection with DeBERTa

A named-entity recognition (NER) pipeline for detecting and redacting **Personally Identifiable Information (PII)** from text using a fine-tuned [`microsoft/deberta-base`](https://huggingface.co/microsoft/deberta-base) model. The project includes synthetic dataset generation, training, evaluation, and inference for redaction.

## Highlights

- **Synthetic data generation** — uses an LLM + [Faker](https://faker.readthedocs.io/) to create realistic PII-rich documents with BIO token-level labels.
- **Fine-tuned DeBERTa** for token classification across 14 PII categories.
- **Redaction inference** — replaces detected PII spans with `xxx` markers.
- **Evaluation** — precision / recall / F1, accuracy, and a confusion matrix.

## PII Categories

| Label | Example |
|-------|---------|
| `CREDIT_CARD` | `4532 0178 9223 5234` |
| `DATE_TIME` | `2024-03-14 09:26:00` |
| `EMAIL_ADDRESS` | `gloria81@example.net` |
| `IBAN_CODE` | `GB82 WEST 1234 5698 7654 32` |
| `IP_ADDRESS` | `61.4.73.188` |
| `NRP` | `American` |
| `LOCATION` | `Petersland` |
| `PERSON` | `Chris Evans` |
| `PHONE_NUMBER` | `+1-283-591-3645x5362` |
| `URL` | `http://mitchell.net/` |
| `US_BANK_NUMBER` | `HQPZ42470309251564` |
| `US_DRIVER_LICENSE` | `1F2A8B` |
| `US_ITIN` | `9-XX-XXXXXX` |
| `US_PASSPORT` | `123456789` |
| `US_SSN` | `078-05-1120` |

## Project Structure

```
.
├── data_prep.py         # Synthetic PII dataset generation (LLM + Faker)
├── get_max_length.py    # Analyze token-sequence lengths to pick max_length
├── train.py             # Fine-tune DeBERTa for token classification
├── test.py              # Evaluate on a held-out test set
├── inference.py         # Load the model and redact PII from input text
├── data/                # Generated datasets (gitignored)
├── results/             # Checkpoints & saved model (gitignored)
└── requirements.txt
```

## Installation

Requires **Python 3.9+**. It is recommended to use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Generate a synthetic dataset

`data_prep.py` calls the OpenAI API to draft documents with PII placeholders, then substitutes them with Faker-generated values and assigns BIO tags. Set your key first:

```bash
export OPENAI_API_KEY="your-key"   # Windows: set OPENAI_API_KEY=your-key
python data_prep.py
```

This writes `data/test_data.json` (and an Excel copy). To generate the training set, increase `num_samples` and adjust the output path in `create_dataset()`.

### 2. Inspect sequence lengths

`get_max_length.py` reports max / average / percentile token lengths so you can tune the `max_length` used during training:

```bash
python get_max_length.py
```

### 3. Train

`train.py` loads `data/train_data.json`, freezes all DeBERTa layers except the final one, and fine-tunes for token classification:

```bash
python train.py
```

The model and tokenizer are saved to `./results/ner_deberta_model_test`. Adjust hyperparameters (epochs, learning rate, batch size) at the top of the file.

### 4. Evaluate

`test.py` evaluates a trained model against `data/test_data.json` and exports a Plotly confusion matrix:

```bash
python test.py
```

### 5. Redact PII

`inference.py` loads a saved model and redacts detected entities from an input document:

```bash
python inference.py
```

## Model & Hyperparameters

- Base model: [`microsoft/deberta-base`](https://huggingface.co/microsoft/deberta-base)
- Task head: `DebertaForTokenClassification`
- `max_length`: `470` (≈ 95th percentile of training sequences — see `get_max_length.py`)
- Learning rate: `2e-5`, batch size: `8`, epochs: `10`
- Fine-tuning: only the last transformer layer (`layer.11`) + head are trainable
- Data split: 80 / 20 train / validation

## Configuration

Most knobs live near the top of each script:

- `train.py` — `max_length`, `num_train_epochs`, `learning_rate`, `per_device_train_batch_size`
- `inference.py` — `skip_list` (label IDs left unredacted), input text
- `data_prep.py` — number of samples, faker replacement mapping

## Notes & Future Work

- The current scripts use relative `./data/...` paths; make sure you run them from the repo root.
- `inference.py`'s redaction logic is intentionally simple (replace with `x`). For production redaction, consider span-based reconstruction across sub-token splits.
- The hardcoded sequence length and `skip_list` should ideally be derived from the trained label map.

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please open an issue or pull request.