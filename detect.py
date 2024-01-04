import warnings
warnings.filterwarnings("ignore")

import os
from tqdm import tqdm
import torch
from glob import glob
import multiprocessing
import json
from utils import labels_to_intervals, SAMPLE_RATE, load_audio
from utils_synth import sound_detector, smooth_sound_detector
import onnxruntime as rt
import numpy as np

#
# Parameters
#

PARAM_MODE = "naive" # "naive" or "neural"
PARAM_SPEECH_DIR_TRAIN = "./dataset/output/speech_train/"
PARAM_SPEECH_DIR_TEST = "./dataset/output/speech_test/"
PARAM_SPEECH_NAIVE_TRESHOLD = 0.01 # Detector sensitivity
PARAM_SPEECH_NEURAL_TRESHOLD = 0.8 # Detector sensitivity
PARAM_SPEECH_NEURAL_ENGINE = "onnx" # "torch" or "onnx"
PARAM_SPEECH_SMOOTH = 8 # Detector smoothing (number of 20ms tokens without voice that could be ignored)
PARAM_WORKERS = multiprocessing.cpu_count()

#
# Loading neural model
#

mel_filters = torch.from_numpy(np.load("./mel_filters.npz", allow_pickle=False)["mel_80"])
if PARAM_MODE == "neural":
    if PARAM_SPEECH_NEURAL_ENGINE == "torch":
        model = torch.jit.load("./detect.pt")
        model.eval()
        # model.to("mps")
    elif PARAM_SPEECH_NEURAL_ENGINE == "onnx":
        onnx_session = rt.InferenceSession("detect.onnx")

#
# Detector
#

def sliding_window(tensor, window_size, step):

    # Load last dimension
    last_dim = tensor.size(-1)
    if window_size > last_dim:
         raise ValueError("Window size is larger than the tensor's last dimension")

    # Create sliding window
    unfolded = tensor.unfold(-1, window_size, step)

    # Permute dimensions
    total_dims = tensor.dim()
    dims = []
    dims.append(total_dims-1)
    for i in range(total_dims - 1):
        dims.append(i)
    dims.append(total_dims)
    unfolded = unfolded.permute(*dims)

    return unfolded

def detect_iter(dir, files, index):
    audio = load_audio(dir + files[index])
    duration = audio.shape[0] / SAMPLE_RATE

    if PARAM_MODE == "naive":
        detected_voice = sound_detector(audio, 320, PARAM_SPEECH_NAIVE_TRESHOLD) # 320 is 20ms
    elif PARAM_MODE == "neural":

        # Pad data
        audio = torch.nn.functional.pad(audio, (3200-320, 0), "constant", 0) # Pad zeros    
        
        # Compute log-mel spectrogram
        window = torch.hann_window(400, device=audio.device)
        stft = torch.stft(audio, 400, 160, window=window, return_complex=False)
        magnitudes = torch.sum((stft ** 2), dim=-1)[..., :-1]
        mel_spec = mel_filters.to(audio.device) @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()


        if PARAM_SPEECH_NEURAL_ENGINE == "torch":
            # detected_voice = model(sliding_window(log_spec.to("mps"), 20, 2)).squeeze()
            # detected_voice = detected_voice > PARAM_SPEECH_NEURAL_TRESHOLD
            predictions = []
            for i in range(3200, len(audio), 320):
                predicted = model(audio[i-3200:i].unsqueeze(0))[0][0]
                predictions.append(1 if predicted > PARAM_SPEECH_NEURAL_TRESHOLD else 0)
            detected_voice = torch.tensor(predictions)
        elif PARAM_SPEECH_NEURAL_ENGINE == "onnx":
            batches = []
            for i in range(20, log_spec.shape[1], 2):
                batches.append(log_spec[:,i-20:i])
            batches = torch.stack(batches)
            predicted = onnx_session.run(["output"],{'input': batches.numpy()})[0]
            detected_voice = (torch.tensor(predicted) > PARAM_SPEECH_NEURAL_TRESHOLD).int().squeeze()
            # predictions = []
            # for i in range(3200, len(audio), 320):
            #     predicted = model(audio[i-3200:i].unsqueeze(0))[0][0]
            #     predictions.append(1 if predicted > PARAM_SPEECH_NEURAL_TRESHOLD else 0)
            # detected_voice = torch.tensor(predictions)
    else:
        raise Exception("Invalid mode")

    # Convert to intervals
    detected_voice = smooth_sound_detector(detected_voice, PARAM_SPEECH_SMOOTH)

    return files[index], duration, labels_to_intervals(detected_voice, 0.02) # 0.02 is 20ms

def detect_parallel(args):
    dir, files, index = args
    return detect_iter(dir, files, index)

def run_detector(dir):

    # Indexing files
    print("Indexing files...")
    files = [os.path.relpath(x, dir) for x in glob(dir + "**/*.wav", recursive=False)]

    # Detector loop
    print("Detecting...")
    output = {}
    total_duration = 0
    total_voice_duration = 0
    total_count = len(files)
    if PARAM_WORKERS == 0:
        for i in tqdm(range(len(files))):
            file, duration, intervals = detect_iter(dir, files, i)
            output[file] = intervals
            total_duration += duration
            for it in intervals:
                total_voice_duration += it[1] - it[0]
    else:
        with multiprocessing.Manager() as manager:
            files = manager.list(files)
            args_list = [(dir, files, i) for i in range(len(files))]
            with multiprocessing.Pool(processes=PARAM_WORKERS) as pool:
                for result in tqdm(pool.imap_unordered(detect_parallel, args_list), total=len(files)):
                    file, duration, intervals = result
                    output[file] = intervals
                    total_duration += duration
                    for it in intervals:
                        total_voice_duration += it[1] - it[0]

    # Save results
    with open(dir + "meta.json", "w") as outfile:
            json.dump(output, outfile)
    # if PARAM_MODE == "naive":
    #     with open(dir + "meta_naive.json", "w") as outfile:
    #         json.dump(output, outfile)
    # if PARAM_MODE == "neural":
    #     if PARAM_SPEECH_NEURAL_ENGINE == "torch":
    #         with open(dir + "meta_torch.json", "w") as outfile:
    #             json.dump(output, outfile)
    #     if PARAM_SPEECH_NEURAL_ENGINE == "onnx":
    #         with open(dir + "meta_onnx.json", "w") as outfile:
    #             json.dump(output, outfile)
    
    # Print results
    print("Total files processed: " + str(total_count))
    print("Total duration: " + str(total_duration))
    print("Total voice duration: " + str(total_voice_duration))
    print("Total voice percentage: " + str(total_voice_duration / total_duration))


#
# Running detector
#

if __name__ == "__main__":
    print("Running detector on test set...")
    run_detector(PARAM_SPEECH_DIR_TEST)
    print("Running detector on train set...")
    run_detector(PARAM_SPEECH_DIR_TRAIN)