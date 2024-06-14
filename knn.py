import os
import numpy as np
import pandas as pd
from skimage import io
from skimage.transform import resize
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

DATASET_DIR = 'data'
IMAGE_SIZE = (80, 80, 3)
NUM_NEIGHBORS = 138

def load_labels(file_name):
    df = pd.read_csv(os.path.join(DATASET_DIR, file_name))
    return dict(zip(df['image_id'], df['label']))

def preprocess_image(image_path):
    image = io.imread(image_path)
    return resize(image, IMAGE_SIZE, anti_aliasing=True).astype(np.float32)

def preprocess_images(image_ids, labels, dataset_type):
    X, y = [], []
    for image_id in image_ids:
        image_path = os.path.join(DATASET_DIR, dataset_type, image_id + '.png')
        X.append(preprocess_image(image_path).flatten())
        y.append(labels[image_id])
    return np.array(X), np.array(y)

def load_and_preprocess_labels():
    train_labels = load_labels('train.csv')
    validation_labels = load_labels('validation.csv')
    return train_labels, validation_labels

def load_and_preprocess_images(train_labels, validation_labels):
    train_df = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
    validation_df = pd.read_csv(os.path.join(DATASET_DIR, 'validation.csv'))
    test_images = os.listdir(os.path.join(DATASET_DIR, 'test'))

    X_train, y_train = preprocess_images(train_df['image_id'], train_labels, 'train')
    X_validation, y_validation = preprocess_images(validation_df['image_id'], validation_labels, 'validation')
    X_test = np.array([preprocess_image(os.path.join(DATASET_DIR, 'test', img)).flatten() for img in test_images])
    
    return X_train, y_train, X_validation, y_validation, X_test, test_images

def train_and_evaluate_knn(X_train, y_train, X_validation, y_validation):
    knn = KNeighborsClassifier(n_neighbors=NUM_NEIGHBORS, metric='manhattan')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_validation)
    accuracy = accuracy_score(y_validation, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    return knn

def make_predictions(knn, X_test, test_images):
    predictions = knn.predict(X_test)
    predictions_df = pd.DataFrame({'image_id': test_images, 'label': predictions})
    predictions_df.to_csv('knn.csv', index=False)
    print("Predictions for test data saved to predictions.csv")

if __name__ == "__main__":
    train_labels, validation_labels = load_and_preprocess_labels()
    X_train, y_train, X_validation, y_validation, X_test, test_images = load_and_preprocess_images(train_labels, validation_labels)
    
    knn = train_and_evaluate_knn(X_train, y_train, X_validation, y_validation)
    make_predictions(knn, X_test, test_images)
