# Image Tagger

A Python application with Gradio UI for batch processing and captioning of images using Qwen2.5-VL vision-language models. The application generates captions that are compatible with Flux image training formats, allowing for easy integration with AI image training workflows.

![Qwen-VL Batch Image Captioner](https://via.placeholder.com/800x400?text=Qwen-VL+Batch+Image+Captioner)

## Features

- Process multiple images in batch with a single operation
- Generate AI-powered image captions using Qwen2.5-VL models
- Support for both 3B (faster) and 7B (more accurate) model variants
- Custom caption prompting for tailored descriptions
- Flux-compatible identifier generation (optional hashed IDs)
- Caption caching for improved performance on repeated operations
- Export captions as a ZIP file of text files (one per image)
- GPU acceleration with CUDA support
- Simple and intuitive Gradio interface

## System Requirements

- **Python**: 3.11.9 or compatible
- **CUDA**: 12.8 (Blackwell compatible)
- **RAM**: 8GB minimum, 16GB+ recommended
- **GPU**: NVIDIA GPU with 6GB+ VRAM recommended for optimal performance
- **Storage**: 10GB+ free space for models and temporary files

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/qwen-vl-batch-captioner.git
cd qwen-vl-batch-captioner
```

2. Create and activate a virtual environment:
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application using the run file to activate the venv and launch app.py::
```bash
run.bat
```

2. Use the interface to:
   - Upload multiple images
   - Select your preferred Qwen-VL model
   - Customize the captioning prompt
   - Set identifier/name for caption prefixing
   - Generate captions and preview them
   - Download captions as a ZIP file

## Quick Start

1. Upload your images using the "Upload Images" section
2. Enter an identifier in the "Identifier or Name" field (e.g., "Joe Smith")
3. Customize the captioning prompt if needed
4. Click "Generate Captions" to process the images
5. Preview the results in the gallery
6. Click "Download Captions ZIP" to export the captions

## How It Works

The application uses Qwen2.5-VL, a vision-language model from Alibaba, to analyze images and generate descriptive captions. The captioning can be guided through custom prompts, allowing you to tailor the descriptions for specific use cases (e.g., describing people, scenes, objects).

For compatibility with Flux image training formats, captions are prefixed with an identifier that can be either the full identifier provided or a hashed format consisting of initials and a short hash (e.g., "Joe Smith" becomes "JS_1a2b3").

## Advanced Usage

### Custom Prompts

The default prompt is designed for describing people in photographs, but you can customize it for different use cases:

- For landscape photography: "Describe this landscape in detail, including terrain, weather, lighting, and mood"
- For product photography: "Describe this product in detail, including its appearance, color, and features"
- For artwork: "Describe this artwork, including style, medium, subject matter, and mood"

### Model Selection

- **Qwen2.5-VL-3B-Instruct**: Faster processing, smaller memory footprint, good for most use cases
- **Qwen2.5-VL-7B-Instruct**: More detailed and accurate captions, requires more VRAM

## Troubleshooting

- **Out of Memory Errors**: Try using the smaller 3B model, or process fewer images at once
- **Slow Processing**: Enable "Use Hash ID" to reduce processing time through caching
- **Missing Dependencies**: Run `pip install -r requirements.txt` to ensure all dependencies are installed
- **CUDA Issues**: Ensure you have compatible NVIDIA drivers installed for CUDA 12.8

## License

MIT License

## Acknowledgments

- This application utilizes the [Qwen2.5-VL](https://github.com/QwenLM/Qwen-VL) model from Alibaba
- Built with [Gradio](https://www.gradio.app/) for the user interface
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) for model integration
