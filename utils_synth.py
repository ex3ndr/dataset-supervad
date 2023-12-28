import random
from .utils import SAMPLE_RATE
import torch
import torchaudio
import torchaudio.functional as F

#
# Applying RIR
#

def reverbrate(waveforms, rir, preprocess = False):
    assert len(waveforms.shape) == 1 # Only single dimension is allowed
    assert len(rir.shape) == 1 # Only single dimension is allowed

    # Find start of a RIR and cut the begining
    if preprocess:
        # Find peak index
        _, direct_index = rir.abs().max(axis=0, keepdim=True)

        # Cut from after peak
        rir = rir[direct_index:]

    # Source length
    source_len = waveforms.shape[0]

    # Normalize remaining
    rir = rir / torch.norm(rir, p=2) # Mean square

    # NOTE: THIS ALL NOT NEEDED for fftconvolve
    # Flip for convolution (we are adding previous values (aka "echo") for each point
    # rir = torch.flip(rir,[0])

    # Pad with zeros to match output time
    # waveforms = torch.cat((torch.zeros(rir.shape[0]-1,dtype=waveforms.dtype), waveforms), 0)

    # Calculate convolution
    waveforms = waveforms.unsqueeze(0).unsqueeze(0)
    rir = rir.unsqueeze(0).unsqueeze(0)
    # waveforms = torch.nn.functional.conv1d(waveforms, rir)
    waveforms = torchaudio.functional.fftconvolve(waveforms, rir)
    waveforms = waveforms.squeeze(dim=0).squeeze(dim=0)
    waveforms = waveforms[0:source_len]

    return waveforms

#
# Naive sound detector
#

def smooth_sound_detector(detections, max_duration):
    assert len(detections.shape) == 1 # Only single dimension is allowed
    output = torch.zeros(detections.shape[0])
    counter = 0
    for i in range(0, detections.shape[0]):
        output[i] = detections[i]
        if detections[i] == 0:
            counter = counter + 1
        else:
            if (counter > 0) and (counter <= max_duration):
                for j in range(i-counter, i):
                    output[j] = 1                    
            counter = 0
    return output

def sound_detector(waveform, frame_size, treshold):
    waveform = waveform.unfold(-1, frame_size, frame_size)
    return (waveform.abs().max(dim=-1).values > treshold).float()

#
# Alligning audio segment to merge into another one
#

def select_random_segment(source, length):
    output = torch.zeros(length)

    # If source is equal to the target
    to_offset = 0
    source_offset = 0
    l = length
    if source.shape[0] < length:  # If source is smaller than needed
        to_offset = random.randint(0, length - source.shape[0])
        l = source.shape[0]
    elif source.shape[0] > length: # IF source is bigger than needed
        source_offset = random.randint(0, source.shape[0] - length)

    # Apply
    output[to_offset:to_offset+l] = output[to_offset:to_offset+l] + source[source_offset:source_offset+l]

    return output
    
#
# Mondifying audio
#

def add_audio_chunk(waveforms, labels, source, speech):
        
    # Calculate offsets
    to_offset = 0
    source_offset = 0
    l = source.shape[0]
    if source.shape[0] < waveforms.shape[0]:
        to_offset = random.randint(0, waveforms.shape[0] - source.shape[0])
        source_offset = 0
        l = source.shape[0]
    elif source.shape[0] > waveforms.shape[0]:
        source_offset = random.randint(0, source.shape[0] - waveforms.shape[0])
        to_offset = 0
        l = waveforms.shape[0]

    # Append
    waveforms[to_offset:to_offset+l] = waveforms[to_offset:to_offset+l] + source[source_offset:source_offset+l]

    # Update labels
    if speech is not None:
        ss = source_offset // 320
        ls = to_offset // 320
        ll = l // 320
        if ll > 0:
            labels[ls : ls + ll] = speech[ss : ss + ll]

def add_audio_noise(waveforms, noise, snr):
    noise = select_random_segment(noise, waveforms.shape[0])
    noise = noise.unsqueeze(0)
    waveforms = waveforms.unsqueeze(0)
    res = F.add_noise(waveforms, noise, torch.tensor([snr]))[0]
    # Sometimes it returns NaN - ignore noise then
    if torch.isnan(res).any():
        return waveforms[0]
    else:
        return res
    
#
# Synthesize sample
#

def synthesize_sample(duration, 
                      
                      # Effector for result
                      effector = None,
                      
                      # Background
                      background = None, 
                      background_snr = 10, # Safe default

                      # Voice
                      clean = None, 
                      clean_treshold = 0.01,
                      clean_smooth = 5, # Default for half of the input sequence or ~100ms
                      clean_tempo = None,

                      # RIR
                      rir = None):
    waveforms = torch.zeros(SAMPLE_RATE * duration)
    labels = torch.zeros(SAMPLE_RATE * duration // 320)

    # Add clean sound
    if clean is not None:

        # Speed up or slow down
        if clean_tempo is not None:
            clean = torchaudio.io.AudioEffector(effect=f'atempo={clean_tempo}').apply(clean.unsqueeze(0).T, SAMPLE_RATE).T[0]

        # Detect voice
        detected_voice = sound_detector(clean, 320, clean_treshold)
        detected_voice = smooth_sound_detector(detected_voice, clean_smooth)

        # Reverbrate. This is a environment feature and we
        # apply it before effects that would simulate voice
        # transmission
        if rir is not None:
            clean = reverbrate(clean, rir)

        # Add audio chunk
        add_audio_chunk(waveforms, labels, clean, detected_voice)

        # Apply background noise
        if background is not None:
            waveforms = add_audio_noise(waveforms, background, background_snr)

        # Apply effector after everything to simluate everything
        if effector is not None:
            waveforms = effector.apply(waveforms.unsqueeze(0).T, SAMPLE_RATE).T[0]

    else:
        # No clean sound: add background as is
        if background is not None:
            add_audio_chunk(waveforms, labels, background, None)

    # Return result
    return waveforms, labels

#
# Effect resolving
#

def resolve(effectOrFunction):
    if isinstance(effectOrFunction, str):
        return effectOrFunction
    elif effectOrFunction is None:
        return None
    else:
        return resolve(effectOrFunction())

def one_of(*args):
    return lambda:resolve(random.choice(list(args)))

def maybe(effect, p):
    return lambda: resolve(effect) if random.random() < p else None

def sequental(*args):
    return lambda: None if (result := ",".join(filter(lambda x: x is not None, map(resolve, args)))) == "" else result