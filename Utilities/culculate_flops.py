from keras_flops import get_flops

model = tf.keras.models.load_model('MODEL_PATH')
model.summary()
# Calculae FLOPS
flops = get_flops(model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")