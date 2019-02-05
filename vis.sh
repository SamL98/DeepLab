root_dir=E:/LERNER/deeplab
script_path=C:/Users/lerner.67a/models/research/deeplab/vis.py

model_name=model-1541525912
#model_name=deeplabv3_pascal_trainval

ckpt_dir="$root_dir/cache_data/$model_name/train"
#ckpt_dir="$root_dir/model_trained/$model_name/"
log_dir="$root_dir/cache_data/$model_name/vis"


#ds_dir="$root_dir/datasets/PascalContext/tfrecord"
ds_dir="D:/datasets/Original/VOCdevkit/tfrecord"


python "$script_path" --logtostderr --vis_split="val" --model_variant="xception_65" --atrous_rates=6 --atrous_rate=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size=513 --vis_crop_size=513 --dataset="pascal_voc_seg" --checkpoint_dir=$ckpt_dir --vis_logdir=$log_dir --dataset_dir=$ds_dir
