"""
To be run on the Raspberry Pi!
```
python scripts/measure/collect_dataset_on_device.py
```

Note that the script is configured for the  Raspberry Pi HQ camera

Parameters set in: configs/collect_dataset.yaml

To test on local machine, set dummy=True (which will just copy the files over).

"""
import glob
import os
import pathlib as plib
import random
import re
import shutil
import tempfile
import time
import warnings
from fractions import Fraction

import cv2
import hydra
import numpy as np
import pygame
import tqdm
from hydra.utils import to_absolute_path
from joblib import Parallel, delayed
from picamera2 import Picamera2
from PIL import Image

from lensless.hardware.constants import (
    RPI_HQ_CAMERA_BLACK_LEVEL,
    RPI_HQ_CAMERA_CCM_MATRIX,
)
from lensless.hardware.slm import adafruit_sub2full, set_programmable_mask
from lensless.utils.image import bayer2rgb_cc, resize
from lensless.utils.io import save_image
from src.scripts.measure.helpers import (
    format_img,
    load_grayscale_video_ffv1,
    patchify_gray_video_np,
    rgb2gray_np,
    save_grayscale_video_ffv1,
)
from src.utils.io_utils import ROOT_PATH


def convert(text):
    return int(text) if text.isdigit() else text.lower()


def alphanum_key(key):
    return [convert(c) for c in re.split("([0-9]+)", key)]


def natural_sort(arr):
    return sorted(arr, key=alphanum_key)


def pre_process_frame(
    frame, pad, vshift, brightness, screen_res, hshift, rot90, landscape, image_res
):
    if image_res is None:
        image_res = (frame.shape[0], frame.shape[1])
    frame = format_img(
        frame,
        pad=pad,
        vshift=vshift,
        brightness=brightness,
        screen_res=screen_res,
        hshift=hshift,
        rot90=rot90,
        landscape=landscape,
        image_res=image_res,
    )  # pre-process
    # formatted images are always RGB
    fd, tmp_path = tempfile.mkstemp(suffix=".npy")
    os.close(fd)  # close the low-level file descriptor
    np.save(tmp_path, frame)
    return tmp_path


@hydra.main(
    version_base=None,
    config_path=str(ROOT_PATH / "src/configs/scripts"),
    config_name="collect_dataset",
)
def collect_dataset(config):
    input_dir = config.input_dir
    output_dir = config.output_dir
    mask_dir = None
    if output_dir is None:
        # create in same directory as input with timestamp
        output_dir = input_dir + "_measured_" + time.strftime("%Y%m%d-%H%M%S")

    # MAX_TRIES = config.max_tries
    # MIN_LEVEL = config.min_level
    # MAX_LEVEL = config.max_level

    # if output dir exists check how many files done
    print(f"Output directory : {output_dir}")
    start_idx = config.start_idx
    if os.path.exists(output_dir):
        files = list(plib.Path(output_dir).glob(f"*.{config.output_file_ext}"))
        n_completed_files = len(files)
        print("\nNumber of completed measurements :", n_completed_files)
        output_dir = plib.Path(output_dir)
        if config.masks is not None:
            mask_dir = plib.Path(output_dir) / "masks"
    else:
        # make output directory
        output_dir = plib.Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        if config.masks is not None:
            # make masks for measurements
            mask_dir = plib.Path(output_dir) / "masks"

            if not mask_dir.exists():
                mask_dir.mkdir(exist_ok=True)

                np.random.seed(config.masks.seed)
                for i in range(config.masks.n):
                    mask_fp = mask_dir / f"mask_{i}.npy"
                    mask_vals = np.random.uniform(0, 1, config.masks.shape)
                    np.save(mask_fp, mask_vals)

    # assert input directory exists
    assert os.path.exists(input_dir)

    # get number of files with glob
    # files = list(plib.Path(input_dir).glob(f"*.{config.input_file_ext}"))
    search_key = f"*{config.input_filter_key}.{config.input_file_ext}"
    files = glob.glob(os.path.join(input_dir, search_key))
    files = natural_sort(files)

    if config.shuffle_files:
        random.seed(config.shuffle_seed)
        random.shuffle(files)

    files = [plib.Path(f) for f in files]
    n_files = len(files)
    print(f"\nNumber of {config.input_file_ext} files :", n_files)
    if config.n_files:
        files = files[: config.n_files]
        n_files = len(files)
        print(f"TEST : collecting first {n_files} files!")

    # initialize screen
    pygame.init()
    screen_res = np.array(config.display.screen_res)  # (width, height)
    # pygame requires W first, then H
    screen = pygame.display.set_mode((screen_res[0], screen_res[1]), pygame.FULLSCREEN)
    # hide cursor
    pygame.mouse.set_visible(False)

    if config.runtime:
        # convert to minutes
        runtime_min = config.runtime * 60
        runtime_sec = runtime_min * 60
        if config.runtime:
            print(f"\nScript will run for (at most) {config.runtime} hour(s).")

    if config.start_delay:
        # wait for this time before starting script
        delay = config.start_delay * 60
        start_time = time.time() + delay
        start_time = time.strftime("%H:%M:%S", time.localtime(start_time))
        print(f"\nScript will start at {start_time}")
        time.sleep(delay)

    print("\nStarting measurement!\n")
    start_time = time.time()

    if not config.dummy:
        res = config.capture.res
        down = config.capture.down

        # set up camera for consistent photos
        # https://picamera.readthedocs.io/en/release-1.13/recipes1.html#capturing-consistent-images
        # https://picamerax.readthedocs.io/en/latest/fov.html?highlight=camera%20resolution#sensor-modes
        # -- just get max resolution of camera
        camera = Picamera2()
        sensor_res = camera.sensor_resolution

        down_res = np.array(sensor_res)
        if down is not None:
            down_res = (np.array(down_res) / down).astype(int)

        if res:
            assert len(res) == 2
        else:
            res = down_res
        camera.close()

        resize_captured = tuple(res) != tuple(down_res)

        # -- now set up camera with desired settings
        max_increase = (
            config.capture.fact_increase * config.max_tries
            if config.max_tries > 0
            else 1
        )
        max_exposure = min(20, config.capture.exposure * max_increase)
        if config.capture.framerate is None:
            framerate = 1 / max_exposure
            warnings.warn(
                f"Framerate is not given. Setting it to 1 / max_exposure = {framerate}"
            )
        elif config.capture.framerate > 1 / max_exposure:
            warnings.warn(
                f"Framerate should be less or equal 1 / max_exposure = {1 / max_exposure}. Resetting framerate"
            )
            framerate = 1 / max_exposure
        else:
            framerate = config.capture.framerate

        camera = Picamera2()
        rgb_conf = {"size": tuple(res), "format": "RGB888"}
        dummy_rgb_conf = {"size": (64, 64), "format": "YUV420"}
        bayer_conf = {"format": "SRGGB12", "size": tuple(sensor_res)}
        still_conf = camera.create_still_configuration(
            main=rgb_conf if config.capture.rgb_mode else dummy_rgb_conf,
            buffer_count=config.capture.buffer_count,
            queue=config.capture.queue,
            raw=None if config.capture.rgb_mode else bayer_conf,
        )
        camera.configure(still_conf)
        print(
            f"Sensor resolution: {sensor_res}, Camera resolution: {res}, down resolution: {down_res}"
        )
        # Wait for the automatic gain control to settle
        camera.start()
        time.sleep(config.capture.config_pause)

        camera_defaults = camera.capture_metadata()
        controls = {
            "AeEnable": False,
            "AwbEnable": False,
            "Contrast": 0.0,
            "Sharpness": 0.0,
            "NoiseReductionMode": 0,
        }
        # framerate
        frame_duration_us = int(1000000 / framerate)
        controls["FrameDurationLimits"] = (frame_duration_us, frame_duration_us)

        # Set ISO to the desired value
        # iso = base_iso * DigitalGain * AnalogueGain
        # base_iso = 100
        # controls["DigitalGain"] = 1
        controls["AnalogueGain"] = config.capture.iso / 100
        # Wait for the automatic gain control to settle
        # Now fix the values

        if config.capture.exposure:
            # in microseconds
            init_shutter_speed = int(config.capture.exposure * 1e6)
        else:
            init_shutter_speed = camera_defaults["ExposureTime"]
        controls["ExposureTime"] = init_shutter_speed

        # AWB
        if config.capture.awb_gains:
            assert len(config.capture.awb_gains) == 2
            g = (
                Fraction(config.capture.awb_gains[0]),
                Fraction(config.capture.awb_gains[1]),
            )
            g = tuple(g)
        else:
            g = camera_defaults["ColourGains"]

        controls["ColourGains"] = g

        camera.set_controls(controls)
        # for parameters to settle
        time.sleep(config.capture.config_pause)

        print("Capturing at resolution: ", res)
        print("AWB gains", float(g[0]), float(g[1]))

    init_brightness = config.display.brightness

    # loop over files with tqdm
    exposure_vals = []
    brightness_vals = []
    # n_tries_vals = []

    # shutter_speed = init_shutter_speed
    data_desc = "Dataset capture"
    for i, _file in enumerate(tqdm.tqdm(files[start_idx:], desc=data_desc)):
        # save file in output directory as MKV
        output_fp = output_dir / _file.name
        output_fp = output_fp.with_suffix(f".{config.output_file_ext}")

        # if not done, perform measurement
        if not os.path.isfile(output_fp):
            if config.dummy:
                shutil.copyfile(_file, output_fp)
                time.sleep(1)
            else:
                video = load_grayscale_video_ffv1(str(_file))
                if config.video.patchify is not None:
                    video = patchify_gray_video_np(
                        video,
                        config.video.patchify.patch_height,
                        config.video.patchify.patch_width,
                    )
                video_len = video.shape[0]
                vid_desc = "Single video capture"
                pre_desc = "Single video preprocessing"
                post_desc = "Single video postprocessing"
                video_frames_paths = []

                # formatting args

                screen_res = np.array(config.display.screen_res)
                hshift = config.display.hshift
                vshift = config.display.vshift
                pad = config.display.pad

                landscape = False
                if config.display.landscape:
                    landscape = True
                image_res = config.display.image_res
                rot90 = config.display.rot90

                # pre-save to avoid extra delays
                video_frames_paths = Parallel(n_jobs=config.n_jobs)(
                    delayed(pre_process_frame)(
                        video[frame_ind],
                        pad=pad,
                        vshift=vshift,
                        brightness=init_brightness,
                        screen_res=screen_res,
                        hshift=hshift,
                        rot90=rot90,
                        landscape=landscape,
                        image_res=image_res,
                    )
                    for frame_ind in tqdm.tqdm(range(video_len), desc=pre_desc)
                )

                output_frame_path_list = []
                output_video_list = []
                for frame_ind in tqdm.tqdm(range(video_len), desc=vid_desc):
                    tmp_path = video_frames_paths[frame_ind]
                    frame = np.load(tmp_path)
                    # display img
                    display_img(frame, screen, config)
                    os.remove(tmp_path)
                    # capture img
                    output, _, _, camera = capture_screen(
                        brightness_vals=brightness_vals,
                        camera=camera,
                        config=config,
                        exposure_vals=exposure_vals,
                        i=i,
                        init_brightness=init_brightness,
                        mask_dir=mask_dir,
                        start_idx=start_idx,
                        filename=output_fp,
                        rgb_mode=config.capture.rgb_mode,
                        down_res=down_res,
                        resize_captured=resize_captured,
                        bayer_res=bayer_conf["size"],
                        first_frame=(frame_ind == 0),
                    )

                    if config.capture.rgb_mode:
                        output_video_list.append(output)
                    else:
                        fd, tmp_path = tempfile.mkstemp(suffix=".npy")
                        os.close(fd)  # close the low-level file descriptor
                        np.save(tmp_path, output)
                        output_frame_path_list.append(tmp_path)

                if not config.capture.rgb_mode:
                    # concat frames into video and save
                    output_video_list = Parallel(n_jobs=config.n_jobs)(
                        delayed(post_process_frame)(frame_path, down, g)
                        for frame_path in tqdm.tqdm(
                            output_frame_path_list, desc=post_desc
                        )
                    )

                output_video = np.stack(output_video_list, axis=0)
                print(f"Max vals: {output_video.max(axis=(1, 2))}")
                save_grayscale_video_ffv1(output_video, str(output_fp))

        # check if runtime is exceeded
        if config.runtime:
            proc_time = time.time() - start_time
            if proc_time > runtime_sec:
                print(f"-- measured {i + 1} / {n_files} files")
                break

    pygame.quit()

    proc_time = time.time() - start_time
    print(f"\nFinished, {proc_time / 60.:.3f} minutes.")

    # print brightness and exposure range and average
    print(f"brightness range: {np.min(brightness_vals)} - {np.max(brightness_vals)}")
    print(f"exposure range: {np.min(exposure_vals)} - {np.max(exposure_vals)}")
    # print(f"n_tries range: {np.min(n_tries_vals)} - {np.max(n_tries_vals)}")
    print(f"brightness average: {np.mean(brightness_vals)}")
    print(f"exposure average: {np.mean(exposure_vals)}")
    # print(f"n_tries average: {np.mean(n_tries_vals)}")


def capture_screen(
    brightness_vals,
    camera,
    config,
    exposure_vals,
    i,
    init_brightness,
    mask_dir,
    start_idx,
    filename,
    rgb_mode=False,
    down_res=[507, 380],
    resize_captured=True,
    bayer_res=[4056, 3040],
    first_frame=True,
):
    if not config.capture.skip:
        # -- set mask pattern
        if mask_dir is not None and first_frame:
            mask_idx = (i + start_idx) % config.masks.n

            # set label for the video
            mask_label = filename.with_suffix(".txt")
            with mask_label.open("w", encoding="utf-8") as text_file:
                text_file.write(str(mask_idx))

            mask_fp = mask_dir / f"mask_{mask_idx}.npy"
            print("using mask: ", mask_fp)
            mask_vals = np.load(mask_fp)
            full_pattern = adafruit_sub2full(
                mask_vals,
                center=config.masks.center,
            )
            set_programmable_mask(full_pattern, device=config.masks.device)
            # give time to set the mask
            time.sleep(config.capture.config_pause)

        # -- take picture
        current_shutter_speed = camera.capture_metadata()["ExposureTime"]

        current_screen_brightness = init_brightness
        print(f"File: {filename}")
        print(f"current shutter speed: {current_shutter_speed}")
        print(f"current screen brightness: {current_screen_brightness}")

        if rgb_mode:
            output = camera.capture_array("main")
            if resize_captured:
                # down_res is wxh
                # cv2 expects wxh
                output = cv2.resize(output, (down_res[0], down_res[1]))
            output = convert_frame_to_grayscale(output)
        else:
            # get bayer data
            raw_data = camera.capture_array("raw")
            output = raw_data.view(np.uint16)
            output = output[:, : bayer_res[0]]  # remove padding

        exposure_vals.append(current_shutter_speed / 1e6)
        brightness_vals.append(current_screen_brightness)
    return output, current_shutter_speed, current_screen_brightness, camera


def display_img(frame, screen, config):
    # frame is HxW
    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    time.sleep(config.display.delay)


def post_process_frame(frame_path, down, g):
    frame = np.load(frame_path)
    # convert to RGB
    output = bayer2rgb_cc(
        frame,
        down=down,
        nbits=12,
        blue_gain=float(g[1]),
        red_gain=float(g[0]),
        black_level=RPI_HQ_CAMERA_BLACK_LEVEL,
        ccm=RPI_HQ_CAMERA_CCM_MATRIX,
        nbits_out=8,
    )

    # convert to grayscale
    output = convert_frame_to_grayscale(output)

    os.remove(frame_path)

    return output


def convert_frame_to_grayscale(frame):
    frame = frame[None, None, ...]  # B x D x H x W x C
    frame = frame.astype(np.float32) / 255  # to float
    frame = rgb2gray_np(frame)  # B x D x H x W x 1
    frame = (frame * 255).astype(np.uint8)  # to uint8
    frame = frame[0, 0, :, :, 0]  # H x W
    return frame


if __name__ == "__main__":
    collect_dataset()
