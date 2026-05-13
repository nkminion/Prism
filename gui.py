import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import gradio as gr
from sklearn.neighbors import NearestNeighbors
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PrismModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.DownBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        self.DownBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
        self.DownBlock3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )
        self.DownBlock4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        )
        self.UpBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        )
        self.UpBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        )
        self.UpBlock3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        self.UpConv4 = nn.Conv2d(in_channels=64, out_channels=313, kernel_size=(3, 3), padding=1)
        self.MaxPool1 = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x1 = self.DownBlock1(x)
        x2 = self.MaxPool1(x1)
        x2 = self.DownBlock2(x2)
        x3 = self.MaxPool1(x2)
        x3 = self.DownBlock3(x3)
        x4 = self.MaxPool1(x3)
        x4 = self.DownBlock4(x4)
        x3 = self.UpBlock1(torch.cat((x4, x3), dim=1))
        x2 = self.UpBlock2(torch.cat((x3, x2), dim=1))
        x1 = self.UpBlock3(torch.cat((x2, x1), dim=1))
        x = self.UpConv4(x1)

        return x


def load_model_and_resources():
    # Load model
    model = PrismModel().to(device)
    model.eval()
    
    try:
        checkpoint = torch.load('PrismModel.pth', map_location=device)
        if isinstance(checkpoint, dict) and 'ModelState' in checkpoint:
            model.load_state_dict(checkpoint['ModelState'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using untrained model")
    
    # Load coordinate buckets for color space mapping
    try:
        coord_buckets = np.load('CoordBuckets.npy')
        nn_model = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(coord_buckets)
        print("Coordinate buckets loaded successfully")
    except Exception as e:
        print(f"Error loading coordinate buckets: {e}")
        coord_buckets = None
        nn_model = None
    
    # Load prior probabilities
    try:
        prior_probs = np.load('PriorProbs.npy')
        print("Prior probabilities loaded successfully")
    except Exception as e:
        print(f"Error loading prior probabilities: {e}")
        prior_probs = None
    
    return model, coord_buckets, nn_model, prior_probs


print("Loading Prism Colorization Model...")
model, coord_buckets, nn_model, prior_probs = load_model_and_resources()


@torch.no_grad()
def colorize_image(input_image):
    """
    Colorize a grayscale or color image.
    
    Args:
        input_image: PIL Image or numpy array
    
    Returns:
        PIL Image (colorized RGB)
    """
    if isinstance(input_image, Image.Image):
        input_array = np.array(input_image)
    else:
        input_array = input_image
    
    # Convert to grayscale if color
    if len(input_array.shape) == 3:
        grayscale = cv2.cvtColor(input_array, cv2.COLOR_RGB2GRAY)
    else:
        grayscale = input_array
    
    # Resize to 256x256
    resized = cv2.resize(grayscale, (256, 256), interpolation=cv2.INTER_CUBIC)
    
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    input_tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(device)
    
    output = model(input_tensor)  # Shape: (1, 313, 256, 256)
    
    color_indices = torch.argmax(output, dim=1).squeeze(0)  # Shape: (256, 256)
    
    if coord_buckets is None or nn_model is None:
        print("Using fallback grayscale output (coordinate buckets not available)")
        colorized_rgb = np.stack([resized, resized, resized], axis=-1)
    else:
        color_indices_flat = color_indices.cpu().numpy().flatten()
        ab_coords = coord_buckets[color_indices_flat]
        ab_coords = ab_coords.reshape(256, 256, 2)
        
        ab_coords = ab_coords + 128.0
        lab_image = np.dstack([resized * 255.0, ab_coords])
        
        # LAB to RGB
        lab_image = lab_image.astype(np.uint8)
        colorized_rgb = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

    return Image.fromarray(colorized_rgb.astype('uint8'))


def gradio_colorize(image):
    try:
        result = colorize_image(image)
        return result
    except Exception as e:
        print(f"Error during colorization: {e}")
        return Image.new('RGB', (256, 256), color='gray')


def create_interface():
    with gr.Blocks(title="Prism - Image Colorization") as demo:
        gr.Markdown("""
        # 🎨 Prism - Image Colorization
        
        Transform grayscale images into vibrant color photos using deep learning!
        
        **How to use:**
        1. Upload a grayscale or color image
        2. The model will convert it to grayscale and predict the colors
        3. View the colorized result
        
        **Model:** PRISM (PyTorch U-Net based colorization model)
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input Image")
                input_img = gr.Image(
                    label="Upload Image (will be converted to grayscale)",
                    type="pil",
                    image_mode="RGB"
                )
                
            with gr.Column():
                gr.Markdown("### Output Image")
                output_img = gr.Image(
                    label="Colorized Output",
                    type="pil"
                )
        
        with gr.Row():
            submit_btn = gr.Button("🎨 Colorize", variant="primary", size="lg")
            clear_btn = gr.Button("Clear", size="lg")
        
        submit_btn.click(
            fn=gradio_colorize,
            inputs=[input_img],
            outputs=[output_img]
        )
        
        clear_btn.click(
            fn=lambda: (None, None),
            outputs=[input_img, output_img]
        )
        
        gr.Markdown("### Try Example (Grayscale)")
        gr.Examples(
            examples=[f"examples/{f}" for f in os.listdir("examples/") if f.endswith((".png", ".jpg", ".jpeg"))] if os.path.isdir("examples/") else [],
            inputs=[input_img],
            outputs=[output_img],
            fn=gradio_colorize,
            cache_examples=False,
        )
        
        gr.Markdown("""
        ---
        ### About
        This model was trained on image colorization using a U-Net architecture with:
        - Grayscale input (L channel)
        - Color prediction in LAB color space (313 discrete color classes)
        - Weighted cross-entropy loss for balanced color prediction
        """)
    
    return demo


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎨 PRISM Image Colorization - Gradio App")
    print("="*60)
    print(f"Device: {device}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print("="*60 + "\n")
    
    demo = create_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
