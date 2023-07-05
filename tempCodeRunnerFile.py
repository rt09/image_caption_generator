generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
# model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
# model.save("models/model_" + str(0) + ".h5")