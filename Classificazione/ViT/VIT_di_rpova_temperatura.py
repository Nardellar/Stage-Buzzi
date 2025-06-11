from transformers import AutoImageProcessor, TFViTForImageClassification
import tensorflow as tf
from datasets import load_dataset

if __name__ == "__main__":

    dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
    image = dataset["test"]["image"][0]

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    inputs = image_processor(image, return_tensors="tf")
    logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = int(tf.math.argmax(logits, axis=-1))
    print(model.config.id2label[predicted_label])
