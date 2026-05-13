import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import gradio as gr
import os

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

TEMPERATURE = 0.38

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
    try:
        buckets = torch.from_numpy(np.load('CoordBuckets.npy')).float().to(DEVICE)
        print("Coordinate buckets loaded successfully")
    except Exception as e:
        print(f"Error loading coordinate buckets: {e}")
        buckets = None

    try:
        checkpoint = torch.load('PrismModel.pth', map_location=DEVICE)
        model = PrismModel().to(DEVICE)
        model.load_state_dict(checkpoint['ModelState'])
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using untrained model")
        model = PrismModel().to(DEVICE)
        model.eval()

    return model, buckets


print("Loading Prism Colorization Model...")
model, buckets = load_model_and_resources()


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

    if len(input_array.shape) == 2:
        input_array = cv2.cvtColor(input_array, cv2.COLOR_GRAY2RGB)
    elif input_array.shape[2] == 4:
        input_array = cv2.cvtColor(input_array, cv2.COLOR_RGBA2RGB)

    h, w = input_array.shape[:2]
    if h < w:
        nw = int(w * (256 / h))
        nh = 256
    else:
        nw = 256
        nh = int(h * (256 / w))

    resized = cv2.resize(input_array, (nw, nh), interpolation=cv2.INTER_AREA)
    start_x = (nw - 256) // 2
    start_y = (nh - 256) // 2
    cropped = resized[start_y:start_y + 256, start_x:start_x + 256]

    lab_image = cv2.cvtColor(cropped, cv2.COLOR_RGB2LAB)
    input_tensor = (torch.from_numpy(lab_image[:, :, 0:1]).float() / 255.0).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    output = model(input_tensor)
    probs = torch.softmax(output / TEMPERATURE, dim=1)

    if buckets is None:
        print("Using fallback grayscale output (coordinate buckets not available)")
        colorized_rgb = cropped
    else:
        ab = torch.einsum('bchw,cd->bdhw', probs, buckets)
        input_tensor *= 255.0
        ab += 128.0
        result = torch.concat((input_tensor, ab), dim=1).squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        result = np.clip(result, 0, 255).astype(np.uint8)
        colorized_rgb = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

    return Image.fromarray(colorized_rgb.astype('uint8'))


def gradio_colorize(image):
    try:
        result = colorize_image(image)
        return result
    except Exception as e:
        print(f"Error during colorization: {e}")
        return Image.new('RGB', (256, 256), color='gray')


def create_interface():
    with gr.Blocks(
        title="Prism - Image Colorization",
        theme=gr.themes.Default(primary_hue="blue")
    ) as demo:
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
    print(f"Device: {DEVICE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print("="*60 + "\n")
    
    demo = create_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
