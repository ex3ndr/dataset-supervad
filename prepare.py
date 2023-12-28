import os
from tqdm import tqdm
from glob import glob
import random
from utils import SAMPLE_RATE, load_audio, save_audio

#
# Parameters
#

MAX_DURATION = 5
MIN_DURATION = 1

#
# File split
#

def split_files(files, to):
    
    # Check directory
    if os.path.isdir(to):
        raise Exception("Directory " + to + " already exist!")
    os.mkdir(to)

    # Process all files
    index = 0
    total_duration = 0
    for i in tqdm(range(len(files))):

        # Load, remove channels and resample if needed
        signal = load_audio(files[i])
        duration = signal.shape[0] / SAMPLE_RATE

        # Skip if too short
        if duration < MIN_DURATION:
            continue

        # Calculate number of splits
        target_duration = duration
        if MAX_DURATION < duration:
            splits = duration // MAX_DURATION
            target_duration = duration / splits

        # Persist audio
        while duration > MAX_DURATION:
            save_audio(to + f'{index:06d}.wav', signal[0:SAMPLE_RATE * MAX_DURATION])
            index = index + 1
            signal = signal[SAMPLE_RATE * MAX_DURATION:]
            duration = signal.shape[0] / SAMPLE_RATE
            total_duration = total_duration + MAX_DURATION
        if duration > MIN_DURATION:
            save_audio(to + f'{index:06d}.wav', signal)
            index = index + 1
            total_duration = total_duration + duration
    
    # Return result
    return index, total_duration

#
# Process RIR
#

def process_impulse(files, to):
    
    # Check directory
    if os.path.isdir(to):
        raise Exception("Directory " + to + " already exist!")
    os.mkdir(to)

    # Process all files
    index = 0
    total_duration = 0
    for i in tqdm(range(len(files))):

        # Load, remove channels and resample if needed
        signal = load_audio(files[i])

        # Find peak index
        value_max, direct_index = signal.abs().max(axis=0, keepdim=True)

        # Cut from after peak
        signal = signal[direct_index:]

        # Limit duration to 2 seconds
        signal = signal[0:SAMPLE_RATE * 2]

        # Calculate duration
        duration = signal.shape[0] / SAMPLE_RATE
        total_duration = total_duration + duration

        # Persist audio
        save_audio(to + f'{index:06d}.wav', signal)
        index = index + 1
    
    # Return result
    return index, total_duration

#
# Listing files
#

def list_all_files(dirs, pattern = "*.wav"):
    wav = []
    for d in dirs:
        wav.extend(glob(d + "**/" + pattern, recursive=True))
    return wav


#
# Directories
#

speech_dirs = [
    "./dataset/source/source_musan/speech/", 
    # "./raw/VOiCES_release/source-16k/train/",
    # "./data/dns_4/datasets_fullband/clean_fullband/emotional_speech/",
    # "./data/dns_4/datasets_fullband/clean_fullband/french_speech/",
    # "./data/dns_4/datasets_fullband/clean_fullband/german_speech/",
    # "./data/dns_4/datasets_fullband/clean_fullband/italian_speech/",
    # "./data/dns_4/datasets_fullband/clean_fullband/read_speech/",
    # "./data/dns_4/datasets_fullband/clean_fullband/russian_speech/",
    # "./data/dns_4/datasets_fullband/clean_fullband/spanish_speech/",
    # "./data/dns_4/datasets_fullband/clean_fullband/vctk_wav48_silence_trimmed/"
]
non_speech_dirs = [
    "./dataset/source/source_musan/music/", 
    "./dataset/source/source_musan/noise/", 
    "./dataset/source/source_rir/pointsource_noises/",

    # "./data/dns_4/datasets_fullband/noise_fullband/",
    # "./raw/VOiCES_release/distant-16k/distractors/",
]
rir_dirs = [
    "./dataset/source/source_rir_sim/",
    # "./raw/VOiCES_release/distant-16k/room-response/rm1/impulse/",
    # "./raw/VOiCES_release/distant-16k/room-response/rm2/impulse/",
    # "./raw/VOiCES_release/distant-16k/room-response/rm3/impulse/",
    # "./raw/VOiCES_release/distant-16k/room-response/rm4/impulse/",
]
rir_special_dirs = [
    "./dataset/source/source_rir/real_rirs_isotropic_noises/",
]

#
# Load and shuffle all files
#

non_speech_files = list_all_files(non_speech_dirs)
speech_files = list_all_files(speech_dirs)
rir_files = list_all_files(rir_dirs) + list_all_files(rir_special_dirs, "*_rir_*.wav")

# Shuffle them in reproducible way
non_speech_files.sort()
speech_files.sort()
rir_files.sort()
rnd = random.Random(42)
rnd.shuffle(speech_files)
rnd.shuffle(non_speech_files)
rnd.shuffle(rir_files)

#
# Do processing
#

# process_impulse(rir_files, "./dataset/output/source_rir/")
split_files(non_speech_files, "./dataset/output/source_non_speech/")
split_files(speech_files, "./dataset/output/source_speech/")