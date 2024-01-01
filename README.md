# 🚀 SuperVAD dataset

This repository contains the one million of 5 second segments of augmented voice and noise combinations with labels.

## Dataset

* Duration: 1M of files, 5 seconds length and 10k of test files.
* Audio format: WAV files with 16kHz sampling rate and 16 bit depth
* Signal-to-noise ratio (SNR): from 3 to 30db
* Source voice speedup or slowdown: from 0.8 to 1.5
* Synthetic and Real Room Impulse Response (RIR) reverberation
* Encoding codecs are included in half of the samples: low/high quality mp3, G2.111

## Downloads

* Training Dataset: https://shared.korshakov.com/datasets/supervad-1/vad_train.tar.gz
* Testing Dataset: https://shared.korshakov.com/datasets/supervad-1/vad_test.tar.gz

# Extra downloads
I am also publishing source files that are used for mixing, they are all wav files withg 16kHz sampling rate:
* Speech Training: https://shared.korshakov.com/datasets/supervad-1/speech_train.tar.gz
* Speech Testing: https://shared.korshakov.com/datasets/supervad-1/speech_test.tar.gz
* Non-Speech files: https://shared.korshakov.com/datasets/supervad-1/non_speech.tar.gz
* Synthetic RIR (cut to the begining and normalized): https://shared.korshakov.com/datasets/supervad-1/rir_synthetic.tar.gz
* Real RIR (cut to the begining and normalized): https://shared.korshakov.com/datasets/supervad-1/rir_real.tar.gz

## References

* [Musan](https://openslr.org/17/) (CC BY 4.0) - Clean Voice and Noises
* [SLR26](https://openslr.org/26/) (CC BY 4.0) - Synthetic RIR
* [SLR28](https://openslr.org/28/) (Apache 2.0) - Real RIR
* [VOiCES](https://iqtlabs.github.io/voices/) (CC BY 4.0) - Clean Voice, Noises and RIR
* [DNS-4](https://github.com/microsoft/DNS-Challenge) (Public Domain/CC BY 4.0/Attr) - Clean Voice and Noises
* [Realistic urban sound mixture dataset](https://zenodo.org/records/1184443) (CC BY 4.0) - Noises
* [Common Voice 16.0](https://commonvoice.mozilla.org/en/datasets) (Mozilla Public License 2.0) - Unused for now

## Reproduction

> [!CAUTION]
> Downloading and synthesizing the dataset requires about 8TB of disk space and several hours to download, unpack and synthesize.

### Downloading sources
To download source datasets, you can invoke `download.sh` script. For this script `aria2` is required.

```bash
./download.sh
```

### Installing dependencies

Script have very limited amount of dependencies that you probabbly already have installed.

```bash
pip install tqdm glob torch torchaudio soundfile
```

### Preparing source datasets

Before synthesizing the dataset, you need to prepare source datasets. To do so, you can invoke `prepare.py` script.

```bash
python3 prepare.py
```

### Synthesizing the dataset

To synthesize the dataset, you can invoke `synthesize.py` script.

```bash
python3 synthesize.py
```

### Packaging the dataset

To package the dataset you need `tar` and `pigz` to be installed.

```bash
./pack.sh
```

# License

CC BY 4.0
