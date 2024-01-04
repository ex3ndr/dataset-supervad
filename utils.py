import soundfile as sf
import librosa
import torch

SAMPLE_RATE = 16000

def intersect_intervals(interval1, interval2):
    """
    Returns the intersection of two intervals, or None if they don't intersect.
    """
    a, b = interval1
    c, d = interval2

    if max(a, c) <= min(b, d):
        return [max(a, c), min(b, d)]
    else:
        return None
    
def labels_to_intervals(labels, scale):
    intervals = []
    start = None

    for i, value in enumerate(labels):
        # Check for the start of a new interval
        if value == 1 and start is None:
            start = i
        # Check for the end of the current interval
        elif value == 0 and start is not None:
            intervals.append((start * scale, (i - 1) * scale))
            start = None

    # Handle the case where the array ends with a 1
    if start is not None:
        intervals.append((start * scale, (len(labels)) * scale))

    return intervals


def load_audio(pathOrTensor):
    y, _ = librosa.load(pathOrTensor, sr=SAMPLE_RATE, mono=True) # I have found that torchaudio sometimes can't open some wav files
    return torch.from_numpy(y)

def save_audio(path, tensor):
    sf.write(path, tensor.numpy(), SAMPLE_RATE, 'PCM_16') # I have found that torchaudio sometimes also corrupts generated wav files