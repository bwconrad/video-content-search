import altair as alt
import cv2
import gradio as gr
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_pil_image, to_tensor


def run(
    path: str,
    model_key: str,
    text_search: str,
    image_search: Image.Image,
    thresh: float,
    stride: int,
    batch_size: int,
    center_crop: bool,
):

    assert path, "An input video should be provided"
    assert (
        text_search is not None or image_search is not None
    ), "A text or image query should be provided"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Initialize model
    name, weights = MODELS[model_key]
    model, _, preprocess = open_clip.create_model_and_transforms(
        name, pretrained=weights, device=device
    )
    model.eval()

    # Remove center crop transform
    if not center_crop:
        del preprocess.transforms[1]

    # Load video
    dataset = LoadVideo(path, transforms=preprocess, vid_stride=stride)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Get text query features
    if text_search:
        # Tokenize search phrase
        tokenizer = open_clip.get_tokenizer(name)
        text = tokenizer([text_search]).to(device)

        # Encode text query
        with torch.no_grad():
            query_features = model.encode_text(text)
            query_features /= query_features.norm(dim=-1, keepdim=True)

    # Get image query features
    else:
        image = preprocess(image_search).unsqueeze(0).to(device)
        with torch.no_grad():
            query_features = model.encode_image(image)
            query_features /= query_features.norm(dim=-1, keepdim=True)

    # Encode each frame and compare with query features
    matches = []
    res = pd.DataFrame(columns=["Frame", "Timestamp", "Similarity"])
    for image, orig, frame, timestamp in dataloader:
        with torch.no_grad():
            image = image.to(device)
            image_features = model.encode_image(image)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        probs = query_features.cpu().numpy() @ image_features.cpu().numpy().T
        probs = probs[0]

        # Save frame similarity values
        df = pd.DataFrame(
            {
                "Frame": frame.tolist(),
                "Timestamp": torch.round(timestamp / 1000, decimals=2).tolist(),
                "Similarity": probs.tolist(),
            }
        )
        res = pd.concat([res, df])

        # Check if frame is over threshold
        for i, p in enumerate(probs):
            if p > thresh:
                matches.append(to_pil_image(orig[i]))

        print(f"Frames: {frame.tolist()} - Probs: {probs}")

    # Create plot of similarity values
    lines = (
        alt.Chart(res)
        .mark_line(color="firebrick")
        .encode(
            alt.X("Timestamp", title="Timestamp (seconds)"),
            alt.Y("Similarity", scale=alt.Scale(zero=False)),
        )
    ).properties(width=600)
    rule = alt.Chart().mark_rule(strokeDash=[6, 3], size=2).encode(y=alt.datum(thresh))

    return matches[:30], lines + rule


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


if __name__ == "__main__":
    text_app = gr.Interface(
        description="Search the content's of a video with a text description.",
        fn=run,
        inputs=[
            gr.Video(label="Video"),
            gr.Dropdown(
                label="Model",
                choices=list(MODELS.keys()),
                value="convnext_base_w - laion2b_s13b_b82k",
            ),
            gr.Textbox(label="Text Search Query"),
            gr.Image(label="Image Search Query", visible=False),
            gr.Slider(label="Threshold", maximum=1.0, value=0.3),
            gr.Slider(label="Frame-rate Stride", value=4, step=1),
            gr.Slider(label="Batch Size", value=4, step=1),
            gr.Checkbox(label="Center Crop"),
        ],
        outputs=[
            gr.Gallery(label="Matched Frames").style(
                columns=2, object_fit="contain", height="auto"
            ),
            gr.Plot(label="Similarity Plot"),
        ],
        allow_flagging="never",
    )

    image_app = gr.Interface(
        description="Search the content's of a video with an image query.",
        fn=run,
        inputs=[
            gr.Video(label="Video"),
            gr.Dropdown(
                label="Model",
                choices=list(MODELS.keys()),
                value="convnext_base_w - laion2b_s13b_b82k",
            ),
            gr.Textbox(label="Text Search Query", visible=False),
            gr.Image(label="Image Search Query", type="pil"),
            gr.Slider(label="Threshold", maximum=1.0, value=0.3),
            gr.Slider(label="Frame-rate Stride", value=4, step=1),
            gr.Slider(label="Batch Size", value=4, step=1),
            gr.Checkbox(label="Center Crop"),
        ],
        outputs=[
            gr.Gallery(label="Matched Frames").style(
                columns=2, object_fit="contain", height="auto"
            ),
            gr.Plot(label="Similarity Plot"),
        ],
        allow_flagging="never",
    )
    app = gr.TabbedInterface(
        interface_list=[text_app, image_app],
        tab_names=["Text Query Search", "Image Query Search"],
        title="CLIP Video Content Search",
    )
    app.launch()
