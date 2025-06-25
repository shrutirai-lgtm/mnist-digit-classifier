from data.data_loader import MNISTDataLoader
from models.cnn_digit_classifier import DigitClassifier


def main():
    data_loader = MNISTDataLoader()
    (x_train, y_train), (x_test, y_test) = data_loader.load_data()

    classifier = DigitClassifier(
        input_shape=data_loader.input_shape,
        num_classes=data_loader.num_classes,
        )
    history = classifier.train(x_train, y_train)

    classifier.evaluate(x_test, y_test)
    classifier.plot_accuracy(history)


if __name__ == "__main__":
    main()