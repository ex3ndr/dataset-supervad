import os
from tqdm import tqdm
from glob import glob
import random
import torch
from multiprocessing import Pool
from utils import SAMPLE_RATE, load_audio, save_audio
import pathlib
import multiprocessing

#
# Parameters
#

PARAM_MAX_DURATION = 5
PARAM_MIN_DURATION = 0.5
PARAM_WORKERS = multiprocessing.cpu_count()

speech_dirs = [
    "./dataset/source/source_musan/speech/", 
    "./dataset/source/source_voices_release/source-16k/train/",
    "./dataset/source/source_dns_challenge_4/clean_fullband/emotional_speech/",
    "./dataset/source/source_dns_challenge_4/clean_fullband/french_speech/",
    "./dataset/source/source_dns_challenge_4/clean_fullband/german_speech/",
    "./dataset/source/source_dns_challenge_4/clean_fullband/italian_speech/",
    "./dataset/source/source_dns_challenge_4/clean_fullband/read_speech/",
    "./dataset/source/source_dns_challenge_4/clean_fullband/russian_speech/",
    "./dataset/source/source_dns_challenge_4/clean_fullband/spanish_speech/",
    "./dataset/source/source_dns_challenge_4/clean_fullband/vctk_wav48_silence_trimmed/",
]
speech_test_dirs = [
    "./dataset/source/source_voices_release/source-16k/test/",
]
non_speech_dirs = [
    "./dataset/source/source_musan/music/", 
    "./dataset/source/source_musan/noise/", 
    # "./dataset/source/source_rir/pointsource_noises/", # This is from Musan too
    "./dataset/source/source_voices_release/distant-16k/distractors/",
    "./dataset/source/source_dns_challenge_4/noise_fullband/",
    "./dataset/source/source_urban_mixture/recordings/"
]

rir_synthetic_dirs = [
    "./dataset/source/source_rir_sim/",
]

rir_dirs = [
    "./dataset/source/source_voices_release/distant-16k/room-response/rm1/impulse/",
    "./dataset/source/source_voices_release/distant-16k/room-response/rm2/impulse/",
    "./dataset/source/source_voices_release/distant-16k/room-response/rm3/impulse/",
    "./dataset/source/source_voices_release/distant-16k/room-response/rm4/impulse/",
]
rir_special_dirs = [
    "./dataset/source/source_rir/real_rirs_isotropic_noises/",
]

#
# File split
#

def split_files_iter(files, to, index, counter, lock):
    # Load, remove channels and resample if needed
    signal = load_audio(files[index])
    duration = signal.shape[0] / SAMPLE_RATE

    # Skip if too short
    if duration < PARAM_MIN_DURATION:
        return None

    # Calculate number of splits
    target_duration = duration
    splits = 1
    if PARAM_MAX_DURATION < duration:
        splits = int(duration // PARAM_MAX_DURATION)
        target_duration = duration / splits

    # Persist audio
    for i in range(splits):

        # Cut audio
        audio = signal
        if i != splits - 1:
            audio = audio[0:int(SAMPLE_RATE * target_duration)]
            signal = signal[int(SAMPLE_RATE * target_duration):]

        # Get file index
        with lock:
            id = counter.value
            counter.value += 1

        # Create dir if needed
        dir = f'{(id// 1000) * 1000:08d}'
        pathlib.Path(to + dir).mkdir(parents=True, exist_ok=True)

        # Save audio
        fname = f'{dir}/{id:08d}.wav'
        save_audio(to + fname, audio)

    return splits, duration

def split_files_parallel(args):
    files, to, index, counter, lock = args
    return split_files_iter(files, to, index, counter, lock)

def split_files(files, to):
    
    # Check directory
    if os.path.isdir(to):
        raise Exception("Directory " + to + " already exist!")
    os.mkdir(to)

    # Process all files
    total_duration = 0
    total_count = 0
    if PARAM_WORKERS == 0:
        counter = multiprocessing.Value('counter', 0)
        lock = manager.Lock()
        for i in tqdm(range(len(files))):
            r = split_files_iter(files, to, i, counter, lock)
            if r is not None:
                splits, duration = r
                total_duration = total_duration + duration
                total_count = total_count + splits
    else:
        with multiprocessing.Manager() as manager:
            files = manager.list(files)
            counter = manager.Value('counter', 0)
            lock = manager.Lock()
            args_list = [(files, to, i, counter, lock) for i in range(len(files))]
            with multiprocessing.Pool(processes=PARAM_WORKERS) as pool:
                for result in tqdm(pool.imap_unordered(split_files_parallel, args_list), total=len(files)):
                    if result is not None:
                        splits, duration = result
                        total_duration = total_duration + duration
                        total_count = total_count + splits
    
    # Return result
    return total_count, total_duration

#
# Process RIR
#

def process_impulse_iter(files, to, index):
    # Load, remove channels and resample if needed
    signal = load_audio(files[index])

    # Find peak index
    _, direct_index = signal.abs().max(axis=0, keepdim=True)

    # Cut from after peak
    signal = signal[direct_index:]

    # Limit duration to 2 seconds
    signal = signal[0:SAMPLE_RATE * 2]

    # Normalize remaining
    signal = signal / torch.norm(signal, p=2) # Mean square

    # Calculate duration
    duration = signal.shape[0] / SAMPLE_RATE

    # Create dir if needed
    dir = f'{(index// 1000) * 1000:08d}'
    pathlib.Path(to + dir).mkdir(parents=True, exist_ok=True)

    # Persist audio
    fname = f'{dir}/{index:08d}.wav'
    save_audio(to + fname, signal)

    return  duration

def process_impulse_parallel(args):
    files, to, i = args
    return process_impulse_iter(files, to, i)

def process_impulse(files, to):
    
    # Check directory
    if os.path.isdir(to):
        raise Exception("Directory " + to + " already exist!")
    pathlib.Path(to).mkdir(parents=True, exist_ok=True)

    # Process all files
    total_duration = 0
    total_count = len(files)
    if PARAM_WORKERS == 0:
        for i in tqdm(range(len(files))):
            total_duration += process_impulse_iter(files, to, i)
    else:
        with multiprocessing.Manager() as manager:
            files = manager.list(files)
            args_list = [(files, to, i) for i in range(len(files))]
            with multiprocessing.Pool(processes=PARAM_WORKERS) as pool:
                for result in tqdm(pool.imap_unordered(process_impulse_parallel, args_list), total=len(files)):
                    total_duration += result
    
    # Return result
    return total_count, total_duration

#
# Listing files
#

def list_all_files(dirs, pattern = "*.wav"):
    wav = []
    for d in dirs:
        wav.extend(glob(d + "**/" + pattern, recursive=True))
    return wav


#
# Main
#

if __name__ == "__main__":
    
    #
    # Load and shuffle all files
    #

    print("Indexing files...")
    non_speech_files = list_all_files(non_speech_dirs)
    speech_files = list_all_files(speech_dirs)
    speech_test_files = list_all_files(speech_test_dirs)
    rir_files = list_all_files(rir_dirs) + \
        list_all_files(rir_special_dirs, "air_type1_air_*.wav") + \
        list_all_files(rir_special_dirs, "RVB2014_type1_rir_*.wav") + \
        list_all_files(rir_special_dirs, "RWCP_type4_rir_????.wav")
    rir_synthetic_files = list_all_files(rir_synthetic_dirs)

    # Shuffle them in reproducible way
    non_speech_files.sort()
    speech_files.sort()
    rir_files.sort()
    rir_synthetic_files.sort()
    speech_test_files.sort()
    rnd = random.Random(42)
    rnd.shuffle(speech_files)
    rnd.shuffle(non_speech_files)
    rnd.shuffle(rir_files)
    rnd.shuffle(speech_test_files)
    rnd.shuffle(rir_synthetic_files)

    #
    # Do processing
    #

    print("Processing Synthetic RIR...")
    i, d = process_impulse(rir_synthetic_files, "./dataset/output/rir_synthetic/")
    print("RIR files: ", i, " duration: ", d)

    print("Processing Real RIR...")
    i, d = process_impulse(rir_files, "./dataset/output/rir_real/")
    print("RIR files: ", i, " duration: ", d)

    print("Processing Non-Speech...")
    i, d = split_files(non_speech_files, "./dataset/output/non_speech/")
    print("Non-Speech files: ", i, " duration: ", d)

    print("Processing Speech...")
    i, d = split_files(speech_files, "./dataset/output/speech_train/")
    print("Speech files: ", i, " duration: ", d)

    print("Processing Speech Test...")
    i, d = split_files(speech_test_files, "./dataset/output/speech_test/")
    print("Speech files: ", i, " duration: ", d)