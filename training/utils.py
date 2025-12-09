import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# FER2013 labels: 0=Angry,1=Disgust,2=Fear,3=Happy,4=Sad,5=Surprise,6=Neutral
TARGET_LABELS = {0: 'angry', 3: 'happy', 4: 'sad', 6: 'neutral'}
ORDERED = ['angry', 'happy', 'sad', 'neutral']
LABEL_TO_IDX = {name: i for i, name in enumerate(ORDERED)}
IDX_TO_LABEL = {i: name for name, i in enumerate(ORDERED)}


def load_fer2013(csv_path: str):
    df = pd.read_csv("C:/Users/prabh/Desktop/Project/Facial Expression Recognition App/fer2013.csv")
    # Filter to target classes only
    df = df[df['emotion'].isin(TARGET_LABELS.keys())].copy()

    # Map to compact indices 0..3 in ORDERED
    df['emotion_name'] = df['emotion'].map(TARGET_LABELS)
    df['label'] = df['emotion_name'].map(LABEL_TO_IDX)

    pixels = df['pixels'].str.split(' ').apply(lambda x: np.array(x, dtype=np.uint8))
    X = np.stack(pixels.to_numpy())
    X = X.reshape((-1, 48, 48, 1)).astype('float32') / 255.0
    y = df['label'].to_numpy()

    # Train/val split with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    return (X_train, y_train), (X_val, y_val)


def build_mobilenet(input_shape=(96, 96, 3), num_classes=4):
    from tensorflow.keras import layers, models, applications

    base = applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights='imagenet'
    )
    base.trainable = False  # freeze for warmup

    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(2.0, offset=-1.0)(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model, base