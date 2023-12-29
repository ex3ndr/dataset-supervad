set -e

# Create folders
DATASETS_PATH="./dataset"
mkdir -p $DATASETS_PATH/archive

# Archive
tar --use-compress-program="pigz --best --recursive" -czvf $DATASETS_PATH/archive/rir_real.tar.gz -C $DATASETS_PATH/output rir_real
tar --use-compress-program="pigz --best --recursive" -czvf $DATASETS_PATH/archive/rir_synthetic.tar.gz -C $DATASETS_PATH/output rir_synthetic
tar --use-compress-program="pigz --best --recursive" -czvf $DATASETS_PATH/archive/speech_train.tar.gz -C $DATASETS_PATH/output speech_train
tar --use-compress-program="pigz --best --recursive" -czvf $DATASETS_PATH/archive/speech_test.tar.gz -C $DATASETS_PATH/output speech_test
tar --use-compress-program="pigz --best --recursive" -czvf $DATASETS_PATH/archive/vad_train.tar.gz -C $DATASETS_PATH/output vad_train
tar --use-compress-program="pigz --best --recursive" -czvf $DATASETS_PATH/archive/vad_test.tar.gz -C $DATASETS_PATH/output vad_test