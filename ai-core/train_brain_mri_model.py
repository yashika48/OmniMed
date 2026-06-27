import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
BACKEND_MODEL_PATH = BASE_DIR.parent / "backend-api" / "brain_tumor_efficientnet.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12


def build_train_generators():
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
    )

    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR / "Training",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
    )

    test_generator = test_datagen.flow_from_directory(
        DATASET_DIR / "Testing",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )

    return train_generator, test_generator


def build_model(num_classes: int) -> Model:
    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.15),
    ])

    x = data_augmentation(inputs)
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=x)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model


def main():
    print('🔎 Building data generators...')
    train_generator, test_generator = build_train_generators()

    class_labels = list(train_generator.class_indices.keys())
    classes = train_generator.classes
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(classes), y=classes)
    class_weights = {i: float(w) for i, w in enumerate(class_weights)}

    for label, weight in zip(class_labels, class_weights.values()):
        print(f'   {label}: class weight={weight:.4f}')

    print('⚙️ Building the EfficientNet model...')
    model = build_model(num_classes=len(class_labels))
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1),
    ]

    print('🚀 Starting training...')
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=test_generator,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    print('✅ Training finished. Evaluating on test set...')
    eval_results = model.evaluate(test_generator, verbose=1)
    print(f'Test loss: {eval_results[0]:.4f}, Test accuracy: {eval_results[1]:.4f}')

    print(f'💾 Saving model to {BACKEND_MODEL_PATH}')
    BACKEND_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(BACKEND_MODEL_PATH, save_format='h5')
    print('✅ Model saved successfully.')


if __name__ == '__main__':
    main()
