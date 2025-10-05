# LenslessMic: Lensless Microphone

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#pre-trained-checkpoints">Pre-trained Checkpoints</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#citation">Citation</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

<p align="center">
<a href="https://arxiv.org/abs/2509.16418">
  <img src="https://img.shields.io/badge/arXiv-2509.16418-b31b1b.svg?logo=arxiv&logoColor=white">
</a>
<a href="https://blinorot.github.io/projects/LenslessMic/">
  <img src="https://img.shields.io/badge/🌐%20Project-Page-green?logo=web&logoColor=white">
</a>
<a href="https://huggingface.co/collections/Blinorot/lenslessmic-68caf4f8ff7fa56c2dac8540">
  <img src="https://img.shields.io/badge/HuggingFace-Collection-yellow.svg?logo=huggingface&logoColor=white">
</a>
<a href="https://github.com/Blinorot/pytorch_project_template/blob/main/LICENSE">
   <img src=https://img.shields.io/badge/license-MIT-blue.svg>
</a>
</p>

## About

This repository contains an official implementation of ["LenslessMic: Audio Encryption and Authentication via Lensless Computational Imaging"](https://arxiv.org/abs/2509.16418).

We represent audio as a time-varying array of images, which is captured by a lensless camera for encryption. Lensless reconstruction algorithms are used to recover audio signal. The method is applicable on different types of audio (speech/music) and different codecs. A codec-agnostic model trained on random data can also be used. LenslessMic serves as a robust audio encryption tool with a physical layer of security and as an authentication methods.

Demo samples are provided on the [project page](https://blinorot.github.io/projects/LenslessMic/) together with additional experiments. Models and datasets are stored in the [HuggingFace Collection](https://huggingface.co/collections/Blinorot/lenslessmic-68caf4f8ff7fa56c2dac8540).

We provide a demo notebook [here](https://github.com/Blinorot/LenslessMic/tree/main/notebooks/Demo.ipynb).

## Installation

Install the environment and dependencies:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)). We used `PYTHON_VERSION=3.11.7`.

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

   Also follow [VISQOL](https://github.com/google/visqol) repo for the metric installation.

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

NeMo ASR toolkit installation:

1. Create new environment:

   ```bash
   conda create --name nemo python==3.11.7
   conda activate nemo
   ```

2. Install required packages:

   ```bash
   pip install "nemo_toolkit[asr]"
   ```

UTMOS installation:

1. Create new environment:

   ```bash
   conda create --name utmos python==3.9.7
   conda activate utmos
   ```

2. Install required packages:

   ```bash
   git clone https://huggingface.co/spaces/sarulab-speech/UTMOS-demo
   cd UTMOS-demo
   pip install -r requirements.txt
   ```

Raspberry Pi installation (for the dataset collection):

0. Install requirements:

   ```bash
   pip install -r rpi_requirements.txt
   ```

1. Install `rawpy` from source:

   ```bash
   git clone https://github.com/LibRaw/LibRaw.git libraw
   git clone https://github.com/LibRaw/LibRaw-cmake.git libraw-cmake
   cd libraw
   git checkout 0.20.0
   cp -R ../libraw-cmake/* .
   cmake .
   sudo make install
   sudo ldconfig

   pip install "cython<3"
   git clone --branch v0.16.0 https://github.com/letmaik/rawpy.git
   cd rawpy
   CFLAGS="-I/usr/local/include" LDFLAGS="-L/usr/local/lib" pip install --no-cache-dir .
   ```

2. Install extra packages:

   ```bash
   pip install git+https://github.com/LCAV/LenslessPiCam.git
   pip install git+https://github.com/pvigier/perlin-numpy.git@5e26837db14042e51166eb6cad4c0df2c1907016
   pip install git+https://github.com/ebezzam/slm-controller.git

   sudo apt install --reinstall libcamera0 python3-libcamera python3-picamera2
   sudo apt install --reinstall libcamera0 libcamera-apps
   # add simlinks to your env
   ln -s /usr/lib/python3/dist-packages/picamera2 ROOT_DIR/LenslessMic/env/lib/python3.9/site-packages/picamera2
   ln -s /usr/lib/python3/dist-packages/picamera2-0.3.12.egg-info ROOT_DIR/ROOT_DIRLenslessMic/env/lib/python3.9/site-packages/
   ln -s /usr/lib/python3/dist-packages/libcamera ROOT_DIR/LenslessMic/env/lib/python3.9/site-packages/libcamera
   ln -s /usr/lib/python3/dist-packages/v4l2.py ROOT_DIR/LenslessMic/env/lib/python3.9/site-packages/v4l2.py
   ln -s /usr/lib/python3/dist-packages/prctl.py ROOT_DIR/LenslessMic/env/lib/python3.9/site-packages/prctl.py
   ln -s /usr/lib/python3/dist-packages/_prctl.cpython-39-arm-linux-gnueabihf.so ROOT_DIR/LenslessMic/env/lib/python3.9/site-packages/
   ln -s /usr/lib/python3/dist-packages/piexif ROOT_DIR/LenslessMic/env/lib/python3.9/site-packages/piexif
   ln -s /usr/lib/python3/dist-packages/pidng ROOT_DIR/LenslessMic/env/lib/python3.9/site-packages/pidng
   ln -s /usr/lib/python3/dist-packages/simplejpeg ROOT_DIR/LenslessMic/env/lib/python3.9/site-packages/simplejpeg
   ln -s /usr/lib/python3/dist-packages/pykms ROOT_DIR/LenslessMic/env/lib/python3.9/site-packages/pykms
   sudo ldconfig
   ```

## Pre-trained checkpoints

To download pre-trained [Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546), use the following command.

```bash
cd scripts
python3 download_dac.py --remote-path ""
```

Add `--download_original` to download original DAC weights. You can download only a specific version of DAC by indicating the path from the [HF repo](https://huggingface.co/Blinorot/dac_finetuned_librispeech), like this:

```bash
cd scripts
python3 download_dac.py --remote-path "16x16_130_16khz/latest/dac/weights.pth"
```

The weights will be saved to `data/dac_exps/`. Use `download_xcodec.py` instead for the [X-Codec](https://arxiv.org/abs/2408.17175).

> [!NOTE]
> The configs for [custom DAC](https://huggingface.co/Blinorot/dac_finetuned_librispeech), fine-tuned on [Librispeech](https://www.openslr.org/12) is located [here](https://github.com/Blinorot/descript-audio-codec).

To download pre-trained LenslessMic models use the following command:

```bash
cd scripts
python3 download_models.py --remote-path ""
```

If you want a specific checkpoint, use its folder name with `/` at the end. For example:

```bash
python3 download_models.py --remote-path "32x32_librispeech_mse_ssim_raw_ssim_PSF_Unet4M_U5_Unet4M/"
```

The description of the models is provided in the model card on [HuggingFace](https://huggingface.co/collections/Blinorot/lenslessmic-68caf4f8ff7fa56c2dac8540).

## Dataset

To download ready-to-use dataset, use the following command:

```bash
cd scripts
python3 download_dataset.py --repo-id "Blinorot/dataset_name" \
   --remote-path "" \
   --local-dir "MANDATORY_LOCAL_DIR"
```

Similar to the models' script, you can download only specific subfolders by setting `remote-path`. You must indicate the local directory. We recommend downloading to `data/datasets/dataset_name`. By default, our code assumes that the dataset names are renamed to:

- `librispeech` for `Blinorot/lensless_mic_librispeech`.
- `songdescriber` for `Blinorot/lensless_mic_songdescriber`.
- `random` for `Blinorot/lensless_mic_random`.

The descriptions of the datasets are provided in the corresponding dataset cards on [HuggingFace](https://huggingface.co/collections/Blinorot/lenslessmic-68caf4f8ff7fa56c2dac8540).

### Dataset collection

To collect data, you need to download DAC/X-Codec weights first via commands written in [Pre-trained checkpoints](#pre-trained-checkpoints) section.

Then, you need to run the following script to save your audio in video format:

```bash
python3 -m src.scripts.processing.convert_dataset dataset.part="test-clean" codec.codec_name="16x16_130_16khz"
```

> [!NOTE]
> Choose another `part` and `codec_name` if you use another codec/partition. Also modify paths below with the corresponding names.

Then, upload it to HF:

```bash
# upload audio
python3 upload_dataset.py --local-dir "PATH_TO_ROOT/data/datasets/librispeech/test-clean/audio" --remote-dir "test-clean/audio" --ignore-unused-audio --ignore-reference-dir "PATH_TO_ROOT/data/datasets/librispeech/test-clean/16x16_130_16khz/lensed/"

# upload video
python3 upload_dataset.py --local-dir "PATH_TO_ROOT/data/datasets/librispeech/test-clean/16x16_130_16khz/lensed" --remote-dir "test-clean/16x16_130_16khz/lensed"
```

Download/Copy your dataset on the Raspberry Pi to an SSD. For example, call it `PATH_TO_RPI_SSD/datasets/librispeech`. Also create a new `tmp_dir` on your SSD. Then, run the following script:

```bash
DISPLAY=:0 TMPDIR=PATH_TO_RPI_SSD/tmp_dir python -m src.scripts.measure.collect_dataset_on_device_v3 -cn=collect_dataset_multimask input_dir=PATH_TO_RPI_SSD/datasets/librispeech/test-clean/16x16_130_16khz/lensed output_dir=PATH_TO_RPI_SSD/datasets/librispeech/test-clean/16x16_130_16khz/lensless_measurement n_files=null
```

If you are generating `train` and `test` sets, ensure to use different random seeds `mask.seed=` and include `mask.reference_dir` to avoid collisions.

Upload collected dataset on HF directly from your RPi.

```bash
cd scripts
python3 upload_dataset.py --local-dir "PATH_TO_PI_SSD/datasets/librispeech/test-clean/16x16_130_16khz/lensless_measurement" --remote-dir "test-clean/16x16_130_16khz/lensless_measurement"
```

> [!IMPORTANT]
> Do not forget to change `username`, `repo-id`, and `paths` to the ones you need. See `scripts` arguments for more details.

## How To Use

We provide an example notebook [here](https://github.com/Blinorot/LenslessMic/tree/main/notebooks/Demo.ipynb).

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments. For example, this code will train `Learned` method from the paper using the default config with CLI modifications:

```bash
python3 train.py trainer.override=True dataloader.train.batch_size=1 \
   dataloader.inference.batch_size=1 \
   writer.run_name=32x32_librispeech_mse_ssim_raw_ssim_PSF_Unet4M_U5_Unet4M \
   codec.codec_name=32x32_120_16khz_original \
   reconstruction=32x32 optimizer.lr=1e-4 \
   loss_function.audio_l1_coef=0 loss_function.raw_codec_ssim_coef=1 \
   loss_function.raw_codec_l1_coef=0 transforms=padcrop_train \
   +loss_function.ssim_kernel=7 +loss_function.ssim_sigma=0.5 \
   +loss_function.raw_ssim_kernel=11
```

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

For example, to run inference for the `Learned` model, run:

```bash
python3 inference.py codec.codec_name=32x32_120_16khz_original \
   reconstruction=32x32 \
   inferencer.model_tag=32x32_librispeech_mse_ssim_raw_ssim_PSF_Unet4M_U5_Unet4M
```

> [!NOTE] > `inference.py` assumes that you moved your model checkpoint to `data/lensless_exps` dir.

By default, the code saves reconstructed audio and video to `data/datasets/reconstructed/{dataset_tag}/{partition}/{model_tag}`.

To calculate metrics, use the `calculate_metrics.py` script. It has the same signature. However, you need to run NeMo first to get ASR transcriptions and speaker embeddings. To do so, use:

```bash
cd scripts
python3 run_asr.py --audio-dir PATH_TO_AUDIO_FILES
python3 run_speaker.py --audio-dir PATH_TO_AUDIO_FILES
```

You can add `--recursive` to apply ASR on all audio files in all subfolders. For UTMOS metric, follow the [UTMOS repository](https://github.com/sarulab-speech/UTMOS22).

We also provide a GAN-based version of the scripts (`train_gan.py`), in which a discriminator-based loss is added to enhance the reconstruction. However, we did not see any improvements using this approach.

## Citation

If you use this repo, please cite it as follows:

```bibtex
@article{grinberg2025lenslessmic,
  title = {LenslessMic: Audio Encryption and Authentication via Lensless Computational Imaging},
  author = {Grinberg, Petr and Bezzam, Eric and Prandoni, Paolo and Vetterli, Martin},
  journal = {arXiv preprint arXiv:2509.16418},
  year = {2025},
}
```

## Credits

This repository is based on [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template). It uses some code from [LenslessPiCam](https://github.com/LCAV/LenslessPiCam) project. We also use [DAC](https://github.com/descriptinc/descript-audio-codec/) and [X-Codec](https://github.com/zhenye234/xcodec) code for neural audio codecs and some of the losses.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
