root_dir=E:/LERNER/deeplab
script_path=C:/Users/lerner.67a/models/research/deeplab/vis.py

model_name="model-pc-1542332575"

ckpt_dir="$root_dir/cache_data/$model_name/train"
log_dir="$root_dir/cache_data/$model_name/vis"

ds_dir="$root_dir/datasets/PascalContext/tfrecord"

python "$script_path" --eval_scales=0.5 --eval_scales=0.75 --eval_scales=1.0 --eval_scales=1.25 --eval_scales=1.5 --eval_scales=1.75 --add_flipped_images=true --logtostderr --vis_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rate=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=513 --vis_crop_size=513 --also_save_raw_predictions=true --dataset="PascalContext" --checkpoint_dir=$ckpt_dir --vis_logdir=$log_dir --dataset_dir=$ds_dir
