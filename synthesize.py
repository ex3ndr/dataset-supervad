import os
from tqdm import tqdm
from glob import glob
import random
import json
import torchaudio
from utils import labels_to_intervals, SAMPLE_RATE, load_audio, save_audio
from utils_synth import synthesize_sample, resolve, sequental, one_of, maybe

#
# Parameters
#

# Synthesizing parameters
PARAM_DURATION = 5
PARAM_COUNT = 1000000

# Speech parameters
PARAM_SPEECH_PROB = 0.5 # Probability of speech presence
PARAM_SPEECH_TRESHOLD = 0.01 # Detector sensitivity
PARAM_SPEECH_SMOOTH = 8 # Detector smoothing (number of 20ms tokens without voice that could be ignored)
PARAM_SPEECH_TEMPO_MIN = 0.8 # Speech tempo min
PARAM_SPEECH_TEMPO_MAX = 1.5 # Speech tempo max

# Background parameters
PARAM_BACKGROUND_PROB = 0.8 # Probability of background presence
PARAM_BACKGROUND_MIN_SNR = 5 # Minimum SNR
PARAM_BACKGROUND_MAX_SNR = 30 # Maximum SNR

# Codec parameters
PARAM_CODECS_PROB = 0.5 # Probability of applying a codec
PARAM_CODECS = [
    {'format': "wav", 'encoder': "pcm_mulaw"},
    {'format': "g722"}, # Popular in VoIP
    # {'format': "ogg", 'encoder': "opus", ""}, # Still experimental?
    {"format": "mp3", "codec_config": torchaudio.io.CodecConfig(bit_rate=8_000)}, # Low quality
    {"format": "mp3", "codec_config": torchaudio.io.CodecConfig(bit_rate=64_000)} # Average quality
]

#
# Synthesizer
#

def synthesize(to):

    # Check directory
    if os.path.isdir(to):
        raise Exception("Directory " + to + " already exist!")
    os.mkdir(to)

    # Speech
    speech_files = glob("./dataset/output/source_speech/**/*.wav", recursive=True)
    background_files = glob("./dataset/output/source_non_speech/**/*.wav", recursive=True)
    rir_files = glob("./dataset/output/source_rir/**/*.wav", recursive=True)

    # RIR
    rir_prob = 0.8

    # Effects
    effects_prob = 0.8
    effects = sequental(
        one_of( # Multiple filters could corrupt audio too badly
            maybe(lambda:f'lowpass=frequency={random.randint(400, 1500)}:poles=1', 0.3),
            maybe(lambda:f'highpass=frequency={random.randint(2000, 4000)}', 0.3),
            maybe("bandpass=frequency=3000", 0.3)
        )
    )

    # Synthesizing loop
    output = {}
    for i in tqdm(range(PARAM_COUNT)):
        # Parts
        clean = None
        background = None
        background_snr = random.uniform(PARAM_BACKGROUND_MIN_SNR, PARAM_BACKGROUND_MAX_SNR)
        rir = None
        effector = None
        speech_tempo = random.uniform(PARAM_SPEECH_TEMPO_MIN, PARAM_SPEECH_TEMPO_MAX)
        codec = None
        effect = None
        effector = None

        # Add speech
        if random.random() < PARAM_SPEECH_PROB:
            clean = load_audio(random.choice(speech_files))
            
        # Add background
        if (random.random() < PARAM_BACKGROUND_PROB or clean is None): # Always pick background if no speech is present
            background = load_audio(random.choice(background_files))

        # Add rir
        if rir_files is not None and random.random() < rir_prob:
            rir = load_audio(random.choice(rir_files))

        # Add codec
        if random.random() < PARAM_CODECS_PROB:
            codec = random.choice(PARAM_CODECS)
            
        # Add effect
        if effects is not None and random.random() < effects_prob:
            effect = resolve(effects)

        # Create effector
        if effect is not None or codec is not None:
            args = {}
            if effect is not None:
                args['effect'] = effect
            if codec is not None:
                args.update(codec)
            effector = torchaudio.io.AudioEffector(**args)

        # Do synthesizing
        sample, labels = synthesize_sample(PARAM_DURATION, 
                                           # Effect
                                           effector=effector,
                                           
                                           # Background
                                           background=background, 
                                           background_snr=background_snr,

                                           # Clean voice
                                           clean=clean,
                                           clean_treshold=PARAM_SPEECH_TRESHOLD,
                                           clean_smooth=PARAM_SPEECH_SMOOTH,
                                           clean_tempo=speech_tempo,

                                           # RIR
                                           rir=rir)

        # Create dir if needed
        dir = f'{(i // 1000) * 1000:08d}'
        if i % 1000 == 0 and not os.path.isdir(to + dir):
            os.mkdir(to + dir)
        fname = f'{dir}/{i:08d}.wav'
        output[fname] = labels_to_intervals(labels)
        save_audio(to + fname, sample)

    # Output
    with open(to + "meta.json", "w") as outfile:
        json.dump(output, outfile)



#
# Synthesize
#
        
synthesize("./dataset/output/synthesized/")