import tensorflow as tf
import pickle
import logging


class TextPredictor:
    def __init__(self, model_path, tokenizer_path):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        """Loads the trained model and tokenizer"""
        try:
            # Load the tokenizer
            with open(self.tokenizer_path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            logging.info("Tokenizer successfully loaded.")

            # Load the trained model
            self.model = tf.keras.models.load_model(self.model_path)
            logging.info("Model successfully loaded.")

        except Exception as e:
            logging.error(f"Error loading model or tokenizer: {str(e)}")
            raise

    def generate_text(self, prompt, max_length=50, temperature=0.7):
        """Generates text based on the prompt"""
        try:
            # Tokenize the prompt
            input_ids = self.tokenizer.encode(prompt, return_tensors='tf')

            # Generate the predicted text
            generated_text = prompt
            for _ in range(max_length):
                # Make a prediction
                predictions = self.model(input_ids)

                # Get the predictions for the last token and apply temperature
                last_token_predictions = predictions[0, -1] / temperature

                # Apply top-k sampling
                top_k = 50
                top_k_indices = tf.argsort(last_token_predictions, direction='DESCENDING')[:top_k]
                top_k_probs = tf.gather(last_token_predictions, top_k_indices)
                top_k_probs = tf.nn.softmax(top_k_probs)

                # Sample from the top-k predictions
                predicted_id = tf.random.categorical(
                    tf.math.log(tf.expand_dims(top_k_probs, 0)),
                    num_samples=1
                )[0, 0]
                predicted_id = top_k_indices[predicted_id]

                # Decode the predicted token
                predicted_word = self.tokenizer.decode(predicted_id)

                # Add the predicted word to the generated text
                generated_text += predicted_word

                # Stop when a sentence-ending punctuation is reached
                if predicted_word.strip() in ['.', '!', '?']:
                    break

                # Prepare new input_ids for the next prediction
                input_ids = self.tokenizer.encode(generated_text, return_tensors='tf')

            return generated_text.strip()

        except Exception as e:
            logging.error(f"Error generating text: {str(e)}")
            return prompt


def main():
    # Model and tokenizer paths
    model_path = 'models/{your_run_folder}/best_model/best_model.keras'
    tokenizer_path = 'tokenizer.pickle'

    # Initialize the TextPredictor
    predictor = TextPredictor(model_path, tokenizer_path)

    # Test prompts for text generation
    test_prompts = [
        "Once upon a time",
        "In a galaxy far, far away",
        "The quick brown fox"
    ]

    logging.info("\nTest results:")
    for prompt in test_prompts:
        generated_text = predictor.generate_text(prompt, max_length=50)
        logging.info(f"\nInput: {prompt}")
        logging.info(f"Generated Output: {generated_text}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
