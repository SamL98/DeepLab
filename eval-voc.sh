root_dir=E:/LERNER/deeplab
script_path=C:/Users/lerner.67a/models/research/deeplab/eval.py

#model_name=model-voc-1541517785

#ckpt_dir="$root_dir/cache_data/$model_name/train"
#log_dir="$root_dir/cache_data/$model_name/eval"

ckpt_dir="$root_dir/model_trained/deeplabv3_pascal_train_aug/"
log_dir="$root_dir/model_trained/deeplabv3_pascal_train_aug/eval"

ds_dir="D:/datasets/Original/VOCdevkit/tfrecord"

python "$script_path" --logtostderr --eval_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rate=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=513 --eval_crop_size=513 --dataset="pascal_voc_seg" --checkpoint_dir=$ckpt_dir --eval_logdir=$log_dir --dataset_dir=$ds_dir
