import torch
from torchviz import make_dot
from torchsummary import summary
import numpy as np
import io
import sys

def visualize_model_architecture(model, save_path="model_architecture.png", txt_path="model_summary.txt",
                                 input_size=(1, 224, 224, 3), use_gpu=False):
    """
    Generates and saves a diagram + text summary of the model architecture.

    Args:
        model: PyTorch model (e.g., SegModel instance).
        save_path: Where to save the diagram (PNG).
        txt_path: Where to save the text summary (TXT).
        input_size: Input tensor size as (batch, H, W, C) because SegModel expects NHWC numpy.
        use_gpu: If True, run on GPU (if available).
    """
    # Create a dummy input in NHWC format
    dummy_input = np.random.randn(*input_size).astype(np.float32)

    # Forward pass to capture graph
    model.eval()
    out_prob, out_labels = model.forward(dummy_input, UseGPU=use_gpu, TrainMode=False)

    # Pick one output tensor to visualize graph
    example_tensor = list(out_prob.values())[0]

    # Generate graph
    dot = make_dot(example_tensor, params=dict(model.named_parameters()))
    dot.format = "png"
    dot.render(save_path.replace(".png", ""))  # saves as .png

    print(f"✅ Model architecture diagram saved to {save_path}")

    # Save textual summary (for PyTorch layers only, expects NCHW input)
    dummy_input_torch = torch.randn(1, 3, 224, 224)  # NCHW for summary

    # Redirect stdout temporarily to capture summary
    buffer = io.StringIO()
    sys.stdout = buffer
    summary(model, (3, 224, 224), device="cuda" if use_gpu else "cpu")
    sys.stdout = sys.__stdout__  # Reset stdout

    # Write captured summary to file
    with open(txt_path, "w") as f:
        f.write(buffer.getvalue())

    print(f"✅ Model summary saved to {txt_path}")


if __name__ == "__main__":
    from model import SegModel
    import CategoryDictionary as CatDic

    net = SegModel(CatDic.CatNum)
    visualize_model_architecture(net,
                                 save_path="segmodel_architecture.png",
                                 txt_path="segmodel_summary.txt",
                                 input_size=(1, 224, 224, 3))
