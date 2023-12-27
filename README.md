# SuperVAD dataset

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

## Download source dataset

Downloading and synthesizing the dataset requires about 2TB of disk space.

```bash
./download.sh
```