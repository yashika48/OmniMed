import numpy as np
from PIL import Image
import io


def get_last_conv_layer(model) -> object:
    """Last top-level Conv2D (kept for flat models / backward compatibility)."""
    for layer in reversed(model.layers):
        if getattr(layer, "__class__", None) and "Conv" in layer.__class__.__name__:
            return layer
    raise ValueError("No Conv2D layer found in model for Grad-CAM generation.")


def normalize_array(array: np.ndarray) -> np.ndarray:
    array = np.maximum(array, 0)
    max_value = np.max(array) if np.max(array) != 0 else 1e-10
    return array / max_value


def _jet_colormap(gray01: np.ndarray) -> np.ndarray:
    """Map a [0,1] heatmap to a jet-style RGB array (no matplotlib dependency)."""
    v = gray01
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def generate_gradcam(model, preprocessed_image: np.ndarray, class_index: int) -> bytes:
    """
    Grad-CAM overlaid on the original scan with a jet colormap.

    Works whether the CNN backbone is NESTED (e.g. EfficientNet saved as a sub-model)
    or flat: for nested models we run the backbone to get its feature map, watch it
    with a GradientTape, then re-apply the head layers — avoiding the 'graph
    disconnected' error that breaks Model()-based Grad-CAM on nested models.
    """
    try:
        import tensorflow as tf
    except ModuleNotFoundError:
        raise RuntimeError("TensorFlow is required to generate Grad-CAM images but is not installed.")

    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break

    x_in = tf.convert_to_tensor(preprocessed_image, dtype=tf.float32)

    if backbone is not None:
        idx = model.layers.index(backbone)
        head_layers = model.layers[idx + 1:]
        with tf.GradientTape() as tape:
            conv_outputs = backbone(x_in, training=False)
            tape.watch(conv_outputs)
            x = conv_outputs
            for layer in head_layers:
                x = layer(x, training=False)
            loss = x[:, class_index]
        grads = tape.gradient(loss, conv_outputs)
    else:
        from tensorflow.keras.models import Model
        last_conv_layer = get_last_conv_layer(model)
        grad_model = Model(model.inputs, [last_conv_layer.output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x_in)
            loss = predictions[:, class_index]
        grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    heatmap = normalize_array(heatmap.numpy())  # small grid, [0,1]

    h, w = preprocessed_image.shape[1], preprocessed_image.shape[2]
    heat_img = Image.fromarray(np.uint8(255 * heatmap)).resize((w, h), Image.BICUBIC)
    heat01 = np.asarray(heat_img, dtype=np.float32) / 255.0

    colored = _jet_colormap(heat01) * 255.0
    scan = preprocessed_image[0].astype(np.float32)  # original resized scan (0-255 RGB)

    # cold areas show the scan; hot areas show color (alpha scales with intensity)
    alpha = (0.55 * heat01)[..., None]
    blended = scan * (1 - alpha) + colored * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    out = io.BytesIO()
    Image.fromarray(blended).save(out, format="PNG")
    return out.getvalue()
