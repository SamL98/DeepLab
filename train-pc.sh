ts=$(date +%s)
root_dir=E:/LERNER/deeplab
script_path=C:/Users/lerner.67a/models/research/deeplab/train.py

ckpt_dir="$root_dir/model_trained/deeplabv3_xception_ade20k_train/model.ckpt"
#ckpt_dir="$root_dir/model_trained/xception_65_coco_pretrained/model.ckpt"
log_dir="$root_dir/cache_data/model-pc-${ts}/train"

ds_dir="$root_dir/datasets/PascalContext/tfrecord"

python "$script_path" --logtostderr --training_number_of_steps=30000 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rate=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size=513 --train_crop_size=513 --train_batch_size=3 --fine_tune_batch_norm=false --initialize_last_layer=false --last_layers_contain_logits_only=false --dataset="PascalContext" --tf_initial_checkpoint=$ckpt_dir --train_logdir=$log_dir --dataset_dir=$ds_dir
