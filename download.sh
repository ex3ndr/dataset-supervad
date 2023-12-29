set -e

# Create folders
OUTPUT_PATH="./dataset"
mkdir -p $OUTPUT_PATH/{source,output}

#
# OpenSLR datasets
#

mkdir -p $OUTPUT_PATH/source/{source_musan,source_rir_sim,source_rir}
wget "https://us.openslr.org/resources/17/musan.tar.gz" -P $OUTPUT_PATH/source
wget "https://us.openslr.org/resources/26/sim_rir_16k.zip" -P $OUTPUT_PATH/source
wget "https://us.openslr.org/resources/28/rirs_noises.zip" -P $OUTPUT_PATH/source
tar -xzvf $OUTPUT_PATH/source/musan.tar.gz -C $OUTPUT_PATH/source/source_musan --strip-components=1
bsdtar -xzvf $OUTPUT_PATH/source/sim_rir_16k.zip -C $OUTPUT_PATH/source/source_rir_sim --strip-components=1
bsdtar -xzvf $OUTPUT_PATH/source/rirs_noises.zip -C $OUTPUT_PATH/source/source_rir --strip-components=1
rm $OUTPUT_PATH/source/musan.tar.gz
rm $OUTPUT_PATH/source/sim_rir_16k.zip
rm $OUTPUT_PATH/source/rirs_noises.zip

#
# Realistic urban sound mixture dataset
#

mkdir -p $OUTPUT_PATH/source/source_urban_mixture
aria2c -x8 --file-allocation=none "https://zenodo.org/records/1184443/files/realistic_urban_sound_mixture_dataset.zip?download=1" -o $OUTPUT_PATH/source/realistic_urban_sound_mixture_dataset.zip
bsdtar -xzvf $OUTPUT_PATH/source/realistic_urban_sound_mixture_dataset.zip -C $OUTPUT_PATH/source/source_urban_mixture --strip-components=1
rm $OUTPUT_PATH/source/realistic_urban_sound_mixture_dataset.zip

#
# Voices datasets
#

mkdir -p $OUTPUT_PATH/source/{source_voices_devkit,source_voices_competition,source_voices_release}

aria2c -x8 --file-allocation=none "https://s3.amazonaws.com/lab41openaudiocorpus/VOiCES_devkit.tar.gz" -o $OUTPUT_PATH/source/voices_devkit.tar.gz
tar -xzvf $OUTPUT_PATH/source/voices_devkit.tar.gz -C $OUTPUT_PATH/source/source_voices_devkit --strip-components=1
rm $OUTPUT_PATH/source/voices_devkit.tar.gz

aria2c -x8 --file-allocation=none "https://s3.amazonaws.com/lab41openaudiocorpus/VOiCES_competition.tar.gz" -o $OUTPUT_PATH/source/voices_competition.tar.gz
tar -xzvf $OUTPUT_PATH/source/voices_competition.tar.gz -C $OUTPUT_PATH/source/source_voices_competition --strip-components=1
rm $OUTPUT_PATH/source/voices_competition.tar.gz

aria2c -x8 --file-allocation=none "https://s3.amazonaws.com/lab41openaudiocorpus/VOiCES_release.tar.gz" -o $OUTPUT_PATH/source/voices_release.tar.gz
tar -xzvf $OUTPUT_PATH/source/voices_release.tar.gz -C $OUTPUT_PATH/source/source_voices_release --strip-components=1
rm $OUTPUT_PATH/source/voices_release.tar.gz

#
# DNS-4 datasets
#

./download_dns4.sh

#
# Common Voice datasets
#

# ./download_cv.sh