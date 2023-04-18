# Video Content Search
An application for searching the content's a video to find frames that match a text or image query. This application utilizes ConvNext CLIP models from [OpenCLIP](https://github.com/mlfoundations/open_clip) to compare video frames with the feature representation of the user's query.

<p align="center">
<img src="assets/text-search-ui.png" width="100%" style={text-align: center;}/>
<img src="assets/image-search-ui.png" width="100%" style={text-align: center;}/>
</p>

## Requirements
- Python 3.8+
- `pip install -r requirements`

## Usage
1. Run `python app.py` and open the given URL in a browser. 
2. Select either the "Text Query Search" or "Image Query Search" tab.
3. Upload your video and write a text query or upload a query image.
4. Adjust any of the parameters.
5. Submit



