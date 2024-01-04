import os
from tqdm import tqdm
from glob import glob
import random
import torch
from multiprocessing import Pool
from utils import SAMPLE_RATE, load_audio, save_audio
import pathlib
import multiprocessing
from prepare_dns import load_dns_noise_with_voice

#
# Parameters
#

PARAM_MAX_DURATION = 5
PARAM_MIN_DURATION = 0.5
PARAM_WORKERS = multiprocessing.cpu_count()

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

def list_all_files(rules):
    wav = []
    for d in rules:
        path = d['path']
        patterns = ["*.wav"]
        if 'patterns' in d:
            patterns = d['patterns']
        for pattern in patterns:
                found = glob(path + "**/" + pattern, recursive=True)
                if 'ignore' in d:
                    filtered = []
                    for f in found:
                        f = f[len(path):]
                        if f not in d['ignore']:
                            filtered.append(path + f)
                        # else:
                        #     print("Ignoring: ", f)
                    found = filtered
                wav.extend(found)
    return wav


#
# Main
#

if __name__ == "__main__":

    #
    # Speech Files
    #

    speech_dirs = [
        { 'path': "./dataset/source/source_musan/speech/" }, 
        { 'path': "./dataset/source/source_voices_release/source-16k/train/" },
        { 'path': "./dataset/source/source_dns_challenge_4/clean_fullband/emotional_speech/" },
        { 'path': "./dataset/source/source_dns_challenge_4/clean_fullband/french_speech/" },
        { 'path': "./dataset/source/source_dns_challenge_4/clean_fullband/german_speech/" },
        { 'path': "./dataset/source/source_dns_challenge_4/clean_fullband/italian_speech/" },
        { 'path': "./dataset/source/source_dns_challenge_4/clean_fullband/read_speech/"  },
        { 'path': "./dataset/source/source_dns_challenge_4/clean_fullband/russian_speech/" },
        { 'path': "./dataset/source/source_dns_challenge_4/clean_fullband/spanish_speech/" },
        { 'path': "./dataset/source/source_dns_challenge_4/clean_fullband/vctk_wav48_silence_trimmed/" },
    ]
    speech_test_dirs = [
        { 'path': "./dataset/source/source_voices_release/source-16k/test/" },
    ]

    #
    # Non-Speech Files
    #

    non_speech_dirs = [

        # Some folders has singing and we are excluging them
        { 'path': "./dataset/source/source_musan/music/fma/" }, 
        { 'path': "./dataset/source/source_musan/music/fma-western-art/" },
        { 'path': "./dataset/source/source_musan/music/hd-classical/" },
        { 'path': "./dataset/source/source_musan/music/rfm/" },
        { 'path': "./dataset/source/source_musan/noise/" }, 

        # { 'path': "./dataset/source/source_rir/pointsource_noises/" }, # This is same from musan
        { 'path': "./dataset/source/source_voices_release/distant-16k/distractors/rm1/babb/" },
        { 'path': "./dataset/source/source_voices_release/distant-16k/distractors/rm1/musi/" },
        { 'path': "./dataset/source/source_voices_release/distant-16k/distractors/rm1/none/" },

        { 'path': "./dataset/source/source_voices_release/distant-16k/distractors/rm2/babb/" },
        { 'path': "./dataset/source/source_voices_release/distant-16k/distractors/rm2/musi/" },
        { 'path': "./dataset/source/source_voices_release/distant-16k/distractors/rm2/none/" },

        { 'path': "./dataset/source/source_voices_release/distant-16k/distractors/rm3/babb/" },
        { 'path': "./dataset/source/source_voices_release/distant-16k/distractors/rm3/musi/" },
        { 'path': "./dataset/source/source_voices_release/distant-16k/distractors/rm3/none/" },

        { 'path': "./dataset/source/source_voices_release/distant-16k/distractors/rm4/babb/" },
        { 'path': "./dataset/source/source_voices_release/distant-16k/distractors/rm4/musi/" },
        { 'path': "./dataset/source/source_voices_release/distant-16k/distractors/rm4/none/" },

        { 'path': "./dataset/source/source_dns_challenge_4/noise_fullband/", 'ignore': load_dns_noise_with_voice() }, # Some noises are with voice
        
        { 'path': "./dataset/source/source_urban_mixture/recordings/" }
    ]

    #
    # Room Impulse Response Files
    #

    rir_synthetic_dirs = [
        { 'path': "./dataset/source/source_rir_sim/" },
    ]
    rir_dirs = [
        { 'path': "./dataset/source/source_voices_release/distant-16k/room-response/rm1/impulse/" },
        { 'path': "./dataset/source/source_voices_release/distant-16k/room-response/rm2/impulse/" },
        { 'path': "./dataset/source/source_voices_release/distant-16k/room-response/rm3/impulse/" },
        { 'path': "./dataset/source/source_voices_release/distant-16k/room-response/rm4/impulse/" },
        { 'path': "./dataset/source/source_rir/real_rirs_isotropic_noises/", 'patterns': ["air_type1_air_*.wav", "RVB2014_type1_rir_*.wav", "RWCP_type4_rir_????.wav"] },
    ]
    
    #
    # Load and shuffle all files
    #

    print("Indexing files...")
    non_speech_files = list_all_files(non_speech_dirs)
    speech_files = list_all_files(speech_dirs)
    speech_test_files = list_all_files(speech_test_dirs)
    rir_files = list_all_files(rir_dirs)
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

    # print("Processing Synthetic RIR...")
    # i, d = process_impulse(rir_synthetic_files, "./dataset/output/rir_synthetic/")
    # print("RIR files: ", i, " duration: ", d)

    # print("Processing Real RIR...")
    # i, d = process_impulse(rir_files, "./dataset/output/rir_real/")
    # print("RIR files: ", i, " duration: ", d)

    print("Processing Non-Speech...")
    i, d = split_files(non_speech_files, "./dataset/output/non_speech/")
    print("Non-Speech files: ", i, " duration: ", d)

    print("Processing Speech...")
    i, d = split_files(speech_files, "./dataset/output/speech_train/")
    print("Speech files: ", i, " duration: ", d)

    print("Processing Speech Test...")
    i, d = split_files(speech_test_files, "./dataset/output/speech_test/")
    print("Speech files: ", i, " duration: ", d)