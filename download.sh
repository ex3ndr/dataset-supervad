set -e

# Create folders
OUTPUT_PATH="./dataset"
mkdir -p $OUTPUT_PATH/{source,output}

# Download musan
wget "https://us.openslr.org/resources/17/musan.tar.gz" -P $OUTPUT_PATH/source
wget "https://us.openslr.org/resources/26/sim_rir_16k.zip" -P $OUTPUT_PATH/source
wget "https://us.openslr.org/resources/28/rirs_noises.zip" -P $OUTPUT_PATH/source

# Extract
tar -xzf $OUTPUT_PATH/source/musan.tar.gz -C $OUTPUT_PATH/source
unzip $OUTPUT_PATH/source/sim_rir_16k.zip -d $OUTPUT_PATH/source
unzip $OUTPUT_PATH/source/rirs_noises.zip -d $OUTPUT_PATH/source