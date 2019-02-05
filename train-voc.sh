ts=$(date +%s)
root_dir=E:/LERNER/deeplab
script_path=C:/Users/lerner.67a/models/research/deeplab/train.py

ckpt_dir="$root_dir/model_trained/deeplabv3_pascal_train_aug/model.ckpt"
log_dir="$root_dir/cache_data/model-voc-${ts}/train"

ds_dir="D:/Datasets/Original/VOCdevkit/tfrecord"

python "$script_path" --logtostderr --training_number_of_steps=1 --train_split="train_aug" --model_variant="xception_65" --atrous_rates=6 --atrous_rate=12 --atrous_rates=18 --atrous_rates=32 --output_stride=16 --decoder_output_stride=4 --train_crop_size=513 --train_crop_size=513 --train_batch_size=3 --fine_tune_batch_norm=false --initialize_last_layer=True --dataset="pascal_voc_seg" --tf_initial_checkpoint=$ckpt_dir --train_logdir=$log_dir --dataset_dir=$ds_dir
