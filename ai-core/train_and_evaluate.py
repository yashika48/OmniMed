"""
Comprehensive Brain MRI Model Training and Evaluation Script
- Trains EfficientNetB0 with class balancing and data augmentation
- Generates confusion matrix and detailed metrics
- Tests on real MRI samples before final deployment
"""
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
BACKEND_MODEL_PATH = BASE_DIR.parent / "backend-api" / "brain_tumor_efficientnet.h5"
OLD_MODEL_PATH = BACKEND_MODEL_PATH.with_stem("brain_tumor_efficientnet_old")
BEST_MODEL_PATH = BASE_DIR / "best_model_temp.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20


def build_train_test_generators():
    """Create data generators for training and testing."""
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
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
    """Build EfficientNetB0-based model with transfer learning."""
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


def get_predictions_and_labels(model, generator):
    """Get predictions and true labels from a generator."""
    predictions = []
    labels = []
    
    for _ in range(len(generator)):
        images, batch_labels = next(generator)
        batch_preds = model.predict(images, verbose=0)
        predictions.append(batch_preds)
        labels.append(batch_labels)
    
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)
    return predictions, labels


def main():
    print('=' * 60)
    print('🧠 BRAIN MRI MODEL TRAINING & EVALUATION')
    print('=' * 60)
    
    print('\n📊 Building data generators...')
    train_generator, test_generator = build_train_test_generators()

    class_labels = list(train_generator.class_indices.keys())
    classes = train_generator.classes
    class_weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(classes), 
        y=classes
    )
    class_weights = {i: float(w) for i, w in enumerate(class_weights)}

    print(f'   Found {len(class_labels)} tumor classes:')
    for label, weight in zip(class_labels, class_weights.values()):
        print(f'   • {label}: weight={weight:.4f}')

    print('\n⚙️  Building EfficientNetB0 model...')
    model = build_model(num_classes=len(class_labels))
    print(f'   Model has {model.count_params():,} total parameters')

    callbacks = [
        ModelCheckpoint(
            str(BEST_MODEL_PATH),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    print('\n🚀 Starting training...')
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=test_generator,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    print('\n📈 Training complete. Loading best model...')
    best_model = tf.keras.models.load_model(BEST_MODEL_PATH)

    print('\n📊 Evaluating on test set...')
    test_loss, test_accuracy = best_model.evaluate(test_generator, verbose=0)
    print(f'   Test Loss: {test_loss:.4f}')
    print(f'   Test Accuracy: {test_accuracy:.4f}')

    print('\n🔍 Generating detailed metrics...')
    test_generator.reset()
    predictions, labels = get_predictions_and_labels(best_model, test_generator)
    
    pred_labels = predictions.argmax(axis=1)
    true_labels = labels.argmax(axis=1)

    # Compute metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

    print(f'\n   📌 Overall Metrics:')
    print(f'      • Accuracy:  {accuracy:.4f}')
    print(f'      • Precision: {precision:.4f}')
    print(f'      • Recall:    {recall:.4f}')
    print(f'      • F1-Score:  {f1:.4f}')

    print(f'\n   📋 Per-Class Metrics:')
    class_report = classification_report(
        true_labels, pred_labels, 
        target_names=class_labels,
        zero_division=0
    )
    print(class_report)

    # Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    print(f'\n   🔲 Confusion Matrix:')
    print(cm)

    # Generate confusion matrix visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap='Blues')
    
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    
    ax.set_title('Confusion Matrix - Test Set')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    fig.colorbar(im, ax=ax, label='Count')
    fig.tight_layout()
    cm_path = BASE_DIR / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=100)
    print(f'   ✅ Saved confusion matrix to {cm_path}')
    plt.close()

    print('\n🖼️  Testing on real MRI samples...')
    test_samples = []
    for class_dir in DATASET_DIR / "Testing" / "*":
        for img_path in list(Path(class_dir).glob("*"))[:2]:  # 2 samples per class
            if img_path.is_file():
                test_samples.append((img_path, class_dir.name))

    if test_samples:
        print(f'   Found {len(test_samples)} test samples')
        sample_correct = 0
        for img_path, true_class in test_samples[:6]:  # Test up to 6 samples
            try:
                from PIL import Image
                image = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
                x = np.expand_dims(np.array(image, dtype=np.float32), 0)
                preds = best_model.predict(x, verbose=0)
                pred_class = class_labels[preds.argmax(axis=1)[0]]
                confidence = preds[0].max()
                is_correct = pred_class == true_class
                sample_correct += is_correct
                
                status = "✅" if is_correct else "❌"
                print(f'   {status} {img_path.name}')
                print(f'      True: {true_class}, Pred: {pred_class} (conf: {confidence:.2%})')
            except Exception as e:
                print(f'   ⚠️  Error processing {img_path.name}: {e}')

        sample_accuracy = sample_correct / min(6, len(test_samples))
        print(f'\n   Sample Accuracy: {sample_accuracy:.2%} ({sample_correct}/{min(6, len(test_samples))})')
    else:
        print('   ⚠️  No test samples found')

    # Compare with old model
    print('\n🔄 Comparing with current production model...')
    new_model_score = accuracy
    old_model_score = None
    
    if BACKEND_MODEL_PATH.exists():
        try:
            old_model = tf.keras.models.load_model(BACKEND_MODEL_PATH)
            test_generator.reset()
            old_preds, old_labels = get_predictions_and_labels(old_model, test_generator)
            old_pred_labels = old_preds.argmax(axis=1)
            old_model_score = accuracy_score(true_labels, old_pred_labels)
            print(f'   Old Model Accuracy: {old_model_score:.4f}')
            print(f'   New Model Accuracy: {new_model_score:.4f}')
            print(f'   Improvement: {(new_model_score - old_model_score):.4f}')
            
            if new_model_score > old_model_score:
                print(f'\n   ✅ New model is BETTER. Replacing production model...')
                if BACKEND_MODEL_PATH.exists():
                    BACKEND_MODEL_PATH.rename(OLD_MODEL_PATH)
                    print(f'   📦 Backed up old model to {OLD_MODEL_PATH}')
                best_model.save(BACKEND_MODEL_PATH, save_format='h5')
                print(f'   💾 Saved new model to {BACKEND_MODEL_PATH}')
            else:
                print(f'\n   ⚠️  New model is NOT better than current. Keeping old model.')
        except Exception as e:
            print(f'   ⚠️  Could not compare with old model: {e}')
    else:
        print(f'   ℹ️  No existing model found. Saving new model as production...')
        best_model.save(BACKEND_MODEL_PATH, save_format='h5')
        print(f'   💾 Saved new model to {BACKEND_MODEL_PATH}')

    # Cleanup
    if BEST_MODEL_PATH.exists():
        BEST_MODEL_PATH.unlink()

    print('\n' + '=' * 60)
    print('✅ TRAINING & EVALUATION COMPLETE')
    print('=' * 60)


if __name__ == '__main__':
    main()
