# ContentGPT - AI Content Generation System

## 📝 Overview
ContentGPT is an advanced AI system that generates contextually relevant content by learning from PDF and CSV files. The system utilizes modern transformer architecture and GPT-2 based language models to produce high-quality text output.

## 🌟 Features

### PDF Processing
- Automatic text extraction from PDF files
- Advanced text cleaning and preprocessing
- Batch processing support
- Configurable minimum line length filtering
- Two processing modes: Simple and Advanced

### CSV Processing
- CSV file data reading and processing
- Flexible column selection
- Automatic text normalization

### Model Features
- GPT-2 based language model
- Bidirectional LSTM layers
- Multi-head attention mechanism
- Transformer blocks with layer normalization
- Customizable model parameters
- Multi-GPU support
- Early stopping and model checkpointing

## 🛠️ Requirements
- Python 3.9
- TensorFlow >= 2.11.0
- Transformers >= 4.29.2
- PyMuPDF >= 1.19.6
- pdfplumber >= 0.8.0
- pandas >= 1.5.2
- tqdm >= 4.64.1
- pickle5 >= 0.0.11

## 📥 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-repo/ContentGPT.git
cd ContentGPT
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### PDF Processing

The system offers two PDF processing modes:

#### 1. Simple Mode
- Place PDF files in the `pdfs` directory
- Run the simple training script:

```bash
python pdf_process/simple/pdf_train_model.py
```

To generate text using the simple model:

```bash
python pdf_process/simple/predict.py
```

#### 2. Advanced Mode
- Place PDF files in the `pdfs` directory
- Update the PDF file path in `pdf_process/advance/advanced_pdf_train_model.py`:

```python
config = {
    'data_path': "../../pdfs/{your_pdf_file}", # Replace with your PDF filename
    ...
}
```

Run the advanced training script:

```bash
python pdf_process/advance/advanced_pdf_train_model.py
```


To generate text using the advanced model:

```bash
python pdf_process/advance/predict.py
```

### CSV Processing

1. Prepare your CSV file with training data
2. Update the CSV file path in `csv_process/advanced__csv_text_model.py`:

```python
config = {
    'data_path': "{your_csv_file}", # Replace with your CSV filename
    'text_column': 'Text', # Replace with your text column name
    ...
}
```

Run the advanced training script:

```bash
python csv_process/advanced/advanced_csv_text_model.py
```

To generate text using the CSV model:

```bash
python csv_process/advanced/predict.py
```


### Text Generation Configuration

You can customize text generation parameters in the predict scripts:

#### Simple PDF Prediction

```python
in pdf_process/simple/predict.py
output = model.generate(
input_ids=inputs['input_ids'],
attention_mask=inputs['attention_mask'],
max_new_tokens=50, # Number of tokens to generate
temperature=0.7, # Controls randomness (0.0-1.0)
top_k=50, # Top K sampling
top_p=0.95, # Nucleus sampling
do_sample=True # Enable sampling
)
```


#### Advanced PDF and CSV Prediction

```python
in pdf_process/advance/predict.py or csv_process/predict.py
generated = predictor.generate_text(
prompt="Your prompt here",
max_length=50, # Maximum length of generated text
temperature=0.7 # Controls randomness (0.0-1.0)
)
```


### Model Paths for Prediction

Make sure to update the model paths in prediction scripts:

#### For CSV Model:

```python
model_path = "models/model_TIMESTAMP/best_model.keras" # Replace TIMESTAMP
tokenizer_path = "tokenizer.pickle"
```


#### For Advanced PDF Model:

```python
model_path = 'models/{your_run_folder}/best_model/best_model.keras'
tokenizer_path = 'tokenizer.pickle'
```


#### For Simple PDF Model:

```python
model_path = 'models/{your_run_folder}/best_model/'
```


## 💾 Memory Management Tips

If you encounter memory issues, try these adjustments:

### Model Configuration

- Adjust `d_model`, `num_heads`, and `dff` in `pdf_process/advance/advanced_pdf_train_model.py` and `csv_process/advanced/advanced_csv_text_model.py` to lower values.
- Decrease `batch_size` in both scripts.
- Increase `validation_split` to 0.2 or higher.
- Increase `early_stopping_patience` to 10 or higher.

### GPU Configuration

- Ensure your GPU has sufficient memory.
- Use mixed precision training if supported by your GPU.

```python
config = {
'max_length': 64, # Reduce from 128 to 64
'batch_size': 4, # Reduce from 8 to 4
'd_model': 512, # Reduce from 768 to 512
'num_heads': 8, # Reduce from 12 to 8
'dff': 2048, # Reduce from 3072 to 2048
}
```


### Memory Optimization Tips:
1. **Batch Size**: Reduce batch size if running out of memory
2. **Sequence Length**: Shorter max_length means less memory usage
3. **Model Size**: Reduce d_model and num_heads for lighter model
4. **GPU Memory**: Enable memory growth:

```python
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```


5. **Data Processing**:
   - Process smaller chunks of data at a time
   - Use generator-based data loading
   - Clear unused variables and call garbage collection

### Training Tips:
- Start with smaller datasets for testing
- Monitor GPU memory usage during training
- Use gradient checkpointing for large models
- Enable mixed precision training for better memory efficiency



## ⚠️ Common Issues and Solutions

1. **Out of Memory Error**
   - Reduce batch size
   - Decrease model size parameters
   - Process smaller text chunks

2. **Slow Training**
   - Enable mixed precision training
   - Optimize data pipeline
   - Use data caching

3. **GPU Issues**
   - Update GPU drivers
   - Enable memory growth
   - Use mixed precision training


This README provides:
- Clear instructions for both PDF processing modes
- Detailed CSV processing setup
- Comprehensive memory management tips
- Common issues and solutions
- Project structure overview
- Installation and usage guidelines