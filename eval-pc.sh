root_dir=E:/LERNER/deeplab
script_path=C:/Users/lerner.67a/models/research/deeplab/eval.py

model_name="model-pc-1542332575"

ckpt_dir="$root_dir/cache_data/$model_name/train"
log_dir="$root_dir/cache_data/$model_name/eval"

ds_dir="$root_dir/datasets/PascalContext/tfrecord"

python "$script_path" --logtostderr --eval_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rate=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size=513 --eval_crop_size=513 --dataset="PascalContext" --checkpoint_dir=$ckpt_dir --eval_logdir=$log_dir --dataset_dir=$ds_dir
