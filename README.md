# 🚀 SuperVAD dataset

This repository contains the one million of 5 second segments of augmented voice and noise combinations with labels.

## Dataset

* Total duration: 5m seconds in 5 seconds intervals
* Audio format: WAV files with 16kHz sampling rate and 16 bit depth
* Signal-to-noise ratio (SNR): from 3 to 30db
* Source voice speedup or slowdown: from 0.8 to 1.5
* Synthetic and Real Room Impulse Response (RIR) reverberation

## Download

TODO

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

# License

CC BY 4.0