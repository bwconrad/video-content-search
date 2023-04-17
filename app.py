import cv2
import gradio as gr
import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import save_image


def run(
    path: str,
    model_key: tuple[str, str],
    search: str,
    thresh: float,
    stride: int,
    batch_size: int,
    center_crop: bool,
):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Initialize model
    name, weights = MODELS[model_key]
    model, _, preprocess = open_clip.create_model_and_transforms(
        name, pretrained=weights, device=device
    )

    # Remove center crop transform
    if not center_crop:
        del preprocess.transforms[1]

    # Tokenize search phrase
    tokenizer = open_clip.get_tokenizer(name)
    text = tokenizer([search]).to(device)

    # Load video
    dataset = LoadVideo(path, transforms=preprocess, vid_stride=stride)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Encode text description once
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Encode each frame and compare with text features
    matches = []
    res = []
    for image, orig, frame, timestamp in dataloader:
        with torch.no_grad():
            image = image.cuda()
            image_features = model.encode_image(image)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        probs = text_features.cpu().numpy() @ image_features.cpu().numpy().T

        for i, p in enumerate(probs[0]):
            if p > thresh:
                matches.append(to_pil_image(orig[i]))

        print(f"Probs: {probs}")

    return matches


class LoadVideo(Dataset):
    def __init__(self, path, transforms, vid_stride=1):

        self.transforms = transforms
        self.vid_stride = vid_stride
        self.cur_frame = 0
        self.cap = cv2.VideoCapture(path)
        self.total_frames = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride
        )

    def __getitem__(self, _):
        # Read video
        # Skip over frames
        for _ in range(self.vid_stride):
            self.cap.grab()

        # Read frame
        _, img = self.cap.retrieve()
        self.cur_frame += 1
        timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC)

        # Convert to PIL
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(np.uint8(img))

        # Apply transforms
        img_t = self.transforms(img)

        return img_t, to_tensor(img), self.cur_frame, timestamp

    def __len__(self):
        return self.total_frames


MODELS = {
    "convnext_base - laion400m_s13b_b51k": ("convnext_base", "laion400m_s13b_b51k"),
    "convnext_base_w - laion2b_s13b_b82k": (
        "convnext_base_w",
        "laion2b_s13b_b82k",
    ),
    "convnext_base_w - laion2b_s13b_b82k_augreg": (
        "convnext_base_w",
        "laion2b_s13b_b82k_augreg",
    ),
    "convnext_base_w - laion_aesthetic_s13b_b82k": (
        "convnext_base_w",
        "laion_aesthetic_s13b_b82k",
    ),
    "convnext_base_w_320 - laion_aesthetic_s13b_b82k": (
        "convnext_base_w_320",
        "laion_aesthetic_s13b_b82k",
    ),
    "convnext_base_w_320 - laion_aesthetic_s13b_b82k_augreg": (
        "convnext_base_w_320",
        "laion_aesthetic_s13b_b82k_augreg",
    ),
    "convnext_large_d - laion2b_s26b_b102k_augreg": (
        "convnext_large_d",
        "laion2b_s26b_b102k_augreg",
    ),
    "convnext_large_d_320 - laion2b_s29b_b131k_ft": (
        "convnext_large_d_320",
        "laion2b_s29b_b131k_ft",
    ),
    "convnext_large_d_320 - laion2b_s29b_b131k_ft_soup": (
        "convnext_large_d_320",
        "laion2b_s29b_b131k_ft_soup",
    ),
    "convnext_xxlarge - laion2b_s34b_b82k_augreg": (
        "convnext_xxlarge",
        "laion2b_s34b_b82k_augreg",
    ),
    "convnext_xxlarge - laion2b_s34b_b82k_augreg_rewind": (
        "convnext_xxlarge",
        "laion2b_s34b_b82k_augreg_rewind",
    ),
    "convnext_xxlarge - laion2b_s34b_b82k_augreg_soup": (
        "convnext_xxlarge",
        "laion2b_s34b_b82k_augreg_soup",
    ),
}


# Run app
description = """
An application for searching the content's of a video with a text query.
"""

app = gr.Interface(
    title="Clip Video Content Search",
    description=description,
    fn=run,
    inputs=[
        gr.Video(label="Video"),
        gr.Dropdown(
            label="Model",
            choices=list(MODELS.keys()),
            value="convnext_base_w - laion2b_s13b_b82k",
        ),
        gr.Textbox(label="Search Query"),
        gr.Slider(label="Threshold", maximum=1.0, value=0.3),
        gr.Slider(label="Frame-rate Stride", value=4, step=1),
        gr.Slider(label="Batch Size", value=4, step=1),
        gr.Checkbox(label="Center Crop"),
    ],
    outputs=gr.Gallery(label="Matched Frames").style(
        columns=2, object_fit="contain", height="auto"
    ),
    allow_flagging="never",
)
app.launch()
