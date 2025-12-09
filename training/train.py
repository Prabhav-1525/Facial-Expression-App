import argparse
import os

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import cv2

from utils import load_fer2013, build_mobilenet, ORDERED


def to_rgb_resized(batch_gray, size=(96, 96)):
    # batch_gray: (N,48,48,1) in [0,1]
    N = batch_gray.shape[0]
    out = np.zeros((N, size[0], size[1], 3), dtype=np.float32)
    for i in range(N):
        g = (batch_gray[i, :, :, 0] * 255).astype(np.uint8)
        g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        out[i] = rgb.astype(np.float32) / 255.0
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='training/fer2013.csv', help='Path to fer2013.csv')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--out', default='models/fer_mnet_4cls.h5')
    parser.add_argument('--unfreeze-at', type=int, default=10, help='Epoch to unfreeze base for fine-tune')
    args = parser.parse_args()

    (X_train_g, y_train), (X_val_g, y_val) = load_fer2013(args.csv)

    # Convert to 3-ch RGB and resize for MobileNetV2
    X_train = to_rgb_resized(X_train_g, (96, 96))
    X_val = to_rgb_resized(X_val_g, (96, 96))

    model, base = build_mobilenet(input_shape=(96, 96, 3), num_classes=len(ORDERED))
    model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Class weights to counter imbalance
    classes = np.arange(len(ORDERED))
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, cw)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cbs = [
        ModelCheckpoint(args.out, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    # Warmup training (frozen base)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max(1, args.unfreeze_at),
        batch_size=args.batch_size,
        shuffle=True,
        verbose=1,
        callbacks=cbs,
        class_weight=class_weights
    )

    # Fine-tune: unfreeze last few layers
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        initial_epoch=max(1, args.unfreeze_at),
        batch_size=args.batch_size,
        shuffle=True,
        verbose=1,
        callbacks=cbs,
        class_weight=class_weights
    )

    print('Saved best model to', args.out)


if __name__ == '__main__':
    main()