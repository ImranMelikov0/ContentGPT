from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import logging

def test_model():
    try:
        # Load model and tokenizer
        model_path = 'models/{your_run_folder}/best_model/'  # Directory where the model is saved
        model = TFGPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)

        # Set padding token and adjust padding direction to the left
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

        # Test input text
        test_text = "Coroutine"

        # Tokenize the text
        inputs = tokenizer(
            test_text,
            return_tensors="tf",  # Return as TensorFlow tensors
            max_length=128,  # Maximum length
            truncation=True,  # Truncate long texts
            padding="max_length"  # Pad to maximum length
        )

        # Make prediction with the model
        output = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=50,  # Only specify the number of new tokens
            num_return_sequences=1,  # Number of sequences to generate
            temperature=0.7,  # Control diversity
            top_k=50,  # Select the top 50 most likely words
            top_p=0.95,  # Limit words to those whose cumulative probability is 95%
            do_sample=True  # Enable sampling mode
        )

        # Decode and print the output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print("Generated text:")
        print(generated_text)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    test_model()
