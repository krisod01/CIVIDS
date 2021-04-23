import tensorflow as tf

for number in range(1,7):

  # Convert the model
  converter = tf.lite.TFLiteConverter.from_saved_model(f"Models/models/model{number}") # path to the SavedModel directory
  tflite_model = converter.convert()

  # Save the model.
  with open(f"Models/tflite_models/model{number}.tflite", "wb") as f:
    f.write(tflite_model)