#  T-Edit: Triple-Branch Diffusion Anchoring for Consistent Editing

---

## 🛠️ Environment Setup

This project builds upon diffusion models for text-to-image editing and requires Python 3.10. We recommend creating a conda environment and installing dependencies as follows:

```bash
conda create --name tedit python=3.10
conda activate tedit
pip install -r requirements.txt
```

## Model Access

T-Edit uses the gated `black-forest-labs/FLUX.1-dev` checkpoint. Before
running the code, accept the model license on its
[Hugging Face page](https://huggingface.co/black-forest-labs/FLUX.1-dev) and
authenticate locally:

```bash
huggingface-cli login
```

By default, the text encoders and tokenizers are loaded from their respective
subfolders in the FLUX repository. You can point to locally downloaded models
and checkpoints with environment variables:

```bash
export TEDIT_T5_MODEL=/path/to/text_encoder_2
export TEDIT_CLIP_MODEL=/path/to/text_encoder
export FLUX_DEV=/path/to/flux1-dev.safetensors
export AE=/path/to/ae.safetensors
```

When these variables are not set, the required files are downloaded from
Hugging Face and cached automatically.

## 🚀 Example Usage: Image Editing

You can edit images using our provided script. Below is an example command to run the editing process:

```python
python edit.py  --source_prompt 'A boy is riding a horse' \
                --target_prompt 'A boy is riding a Camel' \
                --guidance 2 \
                --source_img_dir ./examples/source/horse.jpg \
                --num_steps 30  \
                --name 'flux-dev' --offload \
                --output_dir ./examples/edit/Camel
```


Each run saves a reconstruction as `img_0.jpg` and the edited result as
`img_1.jpg` in the output directory.

### Parameters:

- `--source_prompt`: Description of the source image content (can be left as null if undescribed)
- `--target_prompt`: Desired editing instructions or target description
- `--guidance`: Classifier-free guidance scale controlling prompt adherence (e.g., 2)
- `--source_img_dir`: Path to your source image
- `--num_steps`: Number of diffusion steps (30 recommended)
- `--offload`: Enable model offloading to reduce memory usage
- `--disable_nsfw`: Skip the optional NSFW classifier for offline evaluation
- `--output_dir`: Directory to save edited output images

------

