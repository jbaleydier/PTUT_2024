import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
from sklearn.metrics import confusion_matrix

def charger_images(chemin_base, test_size=0.2):
    images = []
    etiquette = []
    
    for chiffre in range(10):  
        dossier_chiffre = os.path.join(chemin_base, str(chiffre))  
        for i in range(101):  
            chemin_image = os.path.join(dossier_chiffre, f"{i}.jpg")
            if os.path.exists(chemin_image):
                with Image.open(chemin_image) as img:
                    tableau_image = np.asarray(img)
                    tableau_image = tableau_image / 255.0  
                    images.append(tableau_image)  
                    etiquette.append(chiffre)  
    
    images = np.array(images)
    etiquette = np.array(etiquette)

    indices = np.arange(images.shape[0])
    np.random.shuffle(indices)
    images = images[indices]
    etiquette = etiquette[indices]
    
    indice_separation = int(images.shape[0] * (1 - test_size))
    X_train, X_test = images[:indice_separation], images[indice_separation:]
    y_train, y_test = etiquette[:indice_separation], etiquette[indice_separation:]
    
    return X_train, X_test, y_train, y_test

chemin_jeu_donne = "/home/UCA/magilbert10/Dossier/Sujet_Ptut/chiffres/10x10 dataset/"

X_train, X_test, y_train, y_test = charger_images(chemin_jeu_donne, test_size=0.2)

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

class ArretPrecoceAvecPrecision(tf.keras.callbacks.Callback):
    def __init__(self, accuracy=1.0):
        super(ArretPrecoceAvecPrecision, self).__init__()
        self.accuracy = accuracy

    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') >= self.accuracy:
            print(f"\nArrêt de l'entraînement car la précision sur l'ensemble de test atteint {self.accuracy * 100:.2f}%")
            self.model.stop_training = True

# Modèle LSTM
modele = keras.Sequential([
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

modele.save_weights("modele_cnn_weights.h5")
modele.save("modele_cnn_complet.h5")

#modele = keras.models.load_model("modele_cnn_complet.h5")


modele.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

arret_precoce = ArretPrecoceAvecPrecision(accuracy=1.0)

historique = modele.fit(datagen.flow(X_train, y_train, batch_size=100),
                    steps_per_epoch=len(X_train) / 100, epochs=2000000000, validation_data=(X_test, y_test),
                    callbacks=[arret_precoce])

perte_test, precision_test = modele.evaluate(X_test, y_test)
print('\nPrécision sur l\'ensemble de test:', precision_test * 100, '%')

plt.plot(historique.history['accuracy'], label='Précision (entraînement)')
plt.plot(historique.history['val_accuracy'], label='Précision (validation)')
plt.xlabel('Époque')
plt.ylabel('Précision')
plt.legend()
plt.show()

predictions = modele.predict(X_test)
etiquettes_predites = np.argmax(predictions, axis=1)
for chiffre in range(10):
    indices = y_test == chiffre
    predictions_correctes = np.sum(etiquettes_predites[indices] == y_test[indices])
    exemples_totaux = np.sum(indices)
    precision_chiffre = predictions_correctes / exemples_totaux if exemples_totaux > 0 else 0.0
    print(f'Précision pour le chiffre {chiffre}: {precision_chiffre * 100:.2f}%')


historique_perte = historique.history['loss']

plt.plot(historique_perte)
plt.title('Évolution de la perte sur les données d\'entraînement')
plt.xlabel('Répétitions')
plt.ylabel('Perte')
plt.show()

def matrice_confusion(y_vraies, y_predites, classes):
    matrice_confusion = confusion_matrix(y_vraies, y_predites)
    plt.figure(figsize=(10, 7))
    sns.heatmap(matrice_confusion, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraies étiquettes')
    plt.xlabel('Étiquettes prédites')
    plt.show()

matrice_confusion(y_test, etiquettes_predites, classes=[str(i) for i in range(10)])

historique_perte_validation = historique.history['val_loss']

plt.plot(historique_perte_validation)
plt.title('Évolution de la perte sur les données de validation')
plt.xlabel('Répétitions')
plt.ylabel('Perte')
plt.show()