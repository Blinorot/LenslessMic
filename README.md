# LenslessMic: Lensless Microphone

<p align="center">
  <a href="#about">About</a> •
  <a href="#tutorials">Tutorials</a> •
  <a href="#examples">Examples</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#useful-links">Useful Links</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

<p align="center">
<a href="https://github.com/Blinorot/pytorch_project_template/generate">
  <img src="https://img.shields.io/badge/use%20this-template-green?logo=github">
</a>
<a href="https://github.com/Blinorot/pytorch_project_template/blob/main/LICENSE">
   <img src=https://img.shields.io/badge/license-MIT-blue.svg>
</a>
</p>

## About

This repository contains a template for [PyTorch](https://pytorch.org/)-based Deep Learning projects.

The template utilizes different python-dev techniques to improve code readability. Configuration methods enhance reproducibility and experiments control.

The repository is released as a part of the [HSE DLA course](https://github.com/markovka17/dla), however, can easily be adopted for any DL-task.

This template is the official recommended template for the [EPFL CS-433 ML Course](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/).

## Tutorials

This template utilizes experiment tracking techniques, such as [WandB](https://docs.wandb.ai/) and [Comet ML](https://www.comet.com/docs/v2/), and [Hydra](https://hydra.cc/docs/intro/) for the configuration. It also automatically reformats code and conducts several checks via [pre-commit](https://pre-commit.com/). If you are not familiar with these tools, we advise you to look at the tutorials below:

- [Python Dev Tips](https://github.com/ebezzam/python-dev-tips): information about [Git](https://git-scm.com/doc), [pre-commit](https://pre-commit.com/), [Hydra](https://hydra.cc/docs/intro/), and other stuff for better Python code development. The YouTube recording of the workshop is available [here](https://youtu.be/okxaTuBdDuY).

- [Seminar on R&D Coding](https://youtu.be/sEA-Js5ZHxU): Seminar from the [LauzHack Deep Learning Bootcamp](https://github.com/LauzHack/deep-learning-bootcamp/) with template discussion and reasoning. It also explains how to work with [WandB](https://docs.wandb.ai/). The seminar materials can be found [here](https://github.com/LauzHack/deep-learning-bootcamp/blob/main/day03/Seminar_WandB_and_Coding.ipynb).

- [HSE DLA Course Introduction Week](https://github.com/markovka17/dla/tree/2024/week01): combines the two seminars above into one with some updates, including an extra example for [Comet ML](https://www.comet.com/docs/v2/).

- [PyTorch Basics](https://github.com/markovka17/dla/tree/2024/week01/intro_to_pytorch): several notebooks with [PyTorch](https://pytorch.org/docs/stable/index.html) basics and corresponding seminar recordings from the [LauzHack Deep Learning Bootcamp](https://github.com/LauzHack/deep-learning-bootcamp/).

To start working with a template, just click on the `use this template` button.

<a href="https://github.com/Blinorot/pytorch_project_template/generate">
  <img src="https://img.shields.io/badge/use%20this-template-green?logo=github">
</a>

You can choose any of the branches as a starting point. [Set your choice as the default branch](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-branches-in-your-repository/changing-the-default-branch) in the repository settings. You can also [delete unnecessary branches](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository).

## Installation

Installation may depend on your task. The general steps are the following:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

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

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

Raspberry Pi installation:

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

The weights will be saved to `data/dac_exps/`

> [!NOTE]
> The configs for [custom DAC](https://huggingface.co/Blinorot/dac_finetuned_librispeech), fine-tuned on [Librispeech](https://www.openslr.org/12) is located [here](https://github.com/Blinorot/descript-audio-codec).

To download pre-trained LenslessMic models use the following command: TBA.

## Dataset

To download ready-to-use dataset, use the following command:

```bash
cd scripts
python3 download_dataset.py --remote-path "TBA"
```

### Dataset collection

To collect data, you need to download DAC weights first via commands written in [Pre-trained checkpoints](#pre-trained-checkpoints) section.

Then, you need to run the following script to save your audio in video format:

```bash
python3 -m src.scripts.processing.convert_dataset dataset.part="test-clean" codec.codec_name="16x16_130_16khz"
```

Then, upload it to HF:

```bash
# upload audio
python3 upload_dataset.py --local-dir "PATH_TO_ROOT/data/datasets/librispeech/test-clean/audio" --remote-dir "test-clean/audio" --ignore-unused-audio --ignore-reference-dir "PATH_TO_ROOT/data/datasets/librispeech/test-clean/16x16_130_16khz/lensed/"

# upload video
python3 upload_dataset.py --local-dir "PATH_TO_ROOT/data/datasets/librispeech/test-clean/16x16_130_16khz/lensed" --remote-dir "test-clean/16x16_130_16khz/lensed"
```

Download/Copy your dataset on the Raspberry Pi to an SSD. For example, call it `PATH_TO_RPI_SSD/datasets/librispeech`. Also create a new `tmp_dir` on your SSD. Then, run the following script:

```bash
DISPLAY=:0 TMPDIR=PATH_TO_RPI_SSD/tmp_dir python -m src.scripts.measure.collect_dataset_on_device_v3 -cn=collect_dataset_multimask input_dir=PATH_TO_RPI_SSD/datasets/librispeech/test-clean/16x16_130_16khz/lensed output_dir=PATH_TO_RPI_SSD/datasets/librispeech/test-clean/16x16_130_16khz/lenseless_measurement n_files=null
```

Upload collected dataset on HF directly from your RPi.

```bash
cd scripts
python3 upload_dataset.py --local-dir "PATH_TO_PI_SSD/datasets/librispeech/test-clean/16x16_130_16khz/lenseless_measurement" --remote-dir "test-clean/16x16_130_16khz/lenseless_measurement"
```

> [!IMPORTANT]
> Do not forget to change `username`, `repo-id`, and `paths` to the ones you need. See `scripts` arguments for more details.

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

## Useful Links:

You may find the following links useful:

- [Report branch](https://github.com/Blinorot/pytorch_project_template/tree/report): Guidelines for writing a scientific report/paper (with an emphasis on DL projects).

- [CLAIRE Template](https://github.com/CLAIRE-Labo/python-ml-research-template): additional template by [EPFL CLAIRE Laboratory](https://www.epfl.ch/labs/claire/) that can be combined with ours to enhance experiments reproducibility via [Docker](https://www.docker.com/).

- [Mamba](https://github.com/mamba-org/mamba) and [Poetry](https://python-poetry.org/): alternatives to [Conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) and [pip](https://pip.pypa.io/en/stable/installation/) package managers given above.

- [Awesome README](https://github.com/matiassingers/awesome-readme): a list of awesome README files for inspiration. Check the basics [here](https://github.com/PurpleBooth/a-good-readme-template).

## Credits

This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repositories.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
