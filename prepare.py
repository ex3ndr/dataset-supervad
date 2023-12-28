import os
from tqdm import tqdm
from glob import glob
import random
import torch
from multiprocessing import Pool
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
        splits = 1
        if MAX_DURATION < duration:
            splits = int(duration // MAX_DURATION)
            target_duration = duration / splits

        # Persist audio
        for i in range(splits):

            # Cut audio
            audio = signal
            if i != splits - 1:
                audio = audio[0:int(SAMPLE_RATE * target_duration)]
                signal = signal[int(SAMPLE_RATE * target_duration):]

            # Save audio
            dir = f'{(index// 1000) * 1000:08d}'
            if index % 1000 == 0 and not os.path.isdir(to + dir):
                os.mkdir(to + dir)
            fname = f'{dir}/{index:08d}.wav'
            save_audio(to + fname, audio)

            # Update index
            index = index + 1
            total_duration = total_duration + target_duration
    
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
    total_duration = 0
    for i in tqdm(range(len(files))):

        # Load, remove channels and resample if needed
        signal = load_audio(files[i])

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
        total_duration = total_duration + duration

        # Persist audio
        dir = f'{(i// 1000) * 1000:08d}'
        if i % 1000 == 0 and not os.path.isdir(to + dir):
            os.mkdir(to + dir)
        fname = f'{dir}/{i:08d}.wav'
        save_audio(to + fname, signal)
    
    # Return result
    return len(files), total_duration

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
    "./dataset/source/source_voices_release/source-16k/train/",
    # "./data/dns_4/datasets_fullband/clean_fullband/emotional_speech/",
    # "./data/dns_4/datasets_fullband/clean_fullband/french_speech/",
    # "./data/dns_4/datasets_fullband/clean_fullband/german_speech/",
    # "./data/dns_4/datasets_fullband/clean_fullband/italian_speech/",
    # "./data/dns_4/datasets_fullband/clean_fullband/read_speech/",
    # "./data/dns_4/datasets_fullband/clean_fullband/russian_speech/",
    # "./data/dns_4/datasets_fullband/clean_fullband/spanish_speech/",
    # "./data/dns_4/datasets_fullband/clean_fullband/vctk_wav48_silence_trimmed/"
]
speech_test_dirs = [
    "./dataset/source/source_voices_release/source-16k/test/",
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
    # "./dataset/source/source_rir/pointsource_noises/", # This is from Musan too
    "./dataset/source/source_voices_release/distant-16k/distractors/",
    # "./data/dns_4/datasets_fullband/noise_fullband/",
]
rir_dirs = [
    "./dataset/source/source_rir_sim/",
    "./dataset/source/source_voices_release/distant-16k/room-response/rm1/impulse/",
    "./dataset/source/source_voices_release/distant-16k/room-response/rm2/impulse/",
    "./dataset/source/source_voices_release/distant-16k/room-response/rm3/impulse/",
    "./dataset/source/source_voices_release/distant-16k/room-response/rm4/impulse/",
]
rir_special_dirs = [
    "./dataset/source/source_rir/real_rirs_isotropic_noises/",
]

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

# Shuffle them in reproducible way
non_speech_files.sort()
speech_files.sort()
rir_files.sort()
speech_test_files.sort()
rnd = random.Random(42)
rnd.shuffle(speech_files)
rnd.shuffle(non_speech_files)
rnd.shuffle(rir_files)
rnd.shuffle(speech_test_files)

#
# Split to train and test
#

#
# Do processing
#

print("Processing RIR...")
i, d = process_impulse(rir_files, "./dataset/output/rir/")
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