import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
from sklearn.metrics import confusion_matrix


def charger_images(base_path, test_size=0.2):
    images = []
    labels = []
    
    for digit in range(10):  
        digit_folder = os.path.join(base_path, str(digit))  
        for i in range(101):  
            image_path = os.path.join(digit_folder, f"{i}.jpg")
            if os.path.exists(image_path):
                with Image.open(image_path) as img:
                    img_array = np.asarray(img)
                    img_array = img_array / 255.0  
                    images.append(img_array)  
                    labels.append(digit)  
    
    images = np.array(images)
    labels = np.array(labels)

    indices = np.arange(images.shape[0])
    np.random.shuffle(indices)
    images = images[indices]
    labels = labels[indices]
    
    split_idx = int(images.shape[0] * (1 - test_size))
    X_train, X_test = images[:split_idx], images[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]
    
    return X_train, X_test, y_train, y_test

chemin_dataset = "/home/UCA/magilbert10/Dossier/Sujet_Ptut/chiffres/10x10 dataset/"
X_train, X_test, y_train, y_test = charger_images(chemin_dataset, test_size=0.2)

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='nearest'
)
#LSTM modèle ?
model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(10, 10, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(80, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(40, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(datagen.flow(X_train, y_train, batch_size=100),
                    steps_per_epoch=len(X_train) / 100, epochs=1000, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print('\nPrécision sur l\'ensemble de test:', test_acc * 100, '%')

plt.plot(history.history['accuracy'], label='Précision (entraînement)')
plt.plot(history.history['val_accuracy'], label='Précision (validation)')
plt.xlabel('Époque')
plt.ylabel('Précision')
plt.legend()
plt.show()

predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
for chiffre in range(10):
    indices = y_test == chiffre
    correct_predictions = np.sum(predicted_labels[indices] == y_test[indices])
    total_examples = np.sum(indices)
    precision_chiffre = correct_predictions / total_examples if total_examples > 0 else 0.0
    print(f'Précision pour le chiffre {chiffre}: {precision_chiffre * 100:.2f}%')


loss_history = history.history['loss']

# Tracer l'évolution de la perte
plt.plot(loss_history)
plt.title('Evolution de la perte au cours des répétitions')
plt.xlabel('Répétitions')
plt.ylabel('Perte')
plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraies étiquettes')
    plt.xlabel('Étiquettes prédites')
    plt.show()

plot_confusion_matrix(y_test, predicted_labels, classes=[str(i) for i in range(10)])