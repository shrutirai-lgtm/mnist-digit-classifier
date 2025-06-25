import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class DigitClassifier:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    @staticmethod
    def get_model_checkpoint(filename='model.h5', monitor='val_accuracy', mode='max'):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=filename,
            monitor=monitor,
            save_best_only=True,
            save_weights_only=False,
            mode=mode,
            verbose=1
        )

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=self.input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy']
        )

        model.summary()
        return model
    
    def train(self, x_train, y_train, epochs=10, batch_size=64, validation_split=0.1):
        checkpoint_cb = DigitClassifier.get_model_checkpoint()
        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[checkpoint_cb]
        )
        self.model = tf.keras.models.load_model("model.h5")
        return history
    
    def evaluate(self, x_test, y_test):
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        print(f"Test Accuracy: {accuracy:.4f}")
        return loss, accuracy
    
    def predict(self, images):
        height, width, channels = self.input_shape
        if images.ndim == 2:
            images = images.reshape(1, height, width, channels)
        elif images.ndim == 3:
            images = np.expand_dims(images, axis=-1)
        elif images.ndim == 4 and images.shape[-1] != channels:
            raise ValueError(f"Expected channel size {channels}, got {images.shape[-1]}")
        images = images.astype("float32") / 255.0
        probs = self.model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        return preds
    
    def plot_accuracy(self, history):
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training vs Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()
    

    



    

        
    
