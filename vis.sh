root_dir=/Users/samlerner/projects/deeplab
python ~/models/research/deeplab/vis.py \
	--logtostderr \
	--vis_split="val" \
	--model_variant="xception_65" \
	--atrous_rates=6 \
	--atrous_rates=12 \
	--atrous_rates=18 \
	--output_stride=16 \
	--decoder_output_stride=4 \
	--vis_crop_size=400 \
	--vis_crop_size=400 \
	--dataset="PascalContext" \
	--checkpoint_dir="$root_dir/model_trained/deeplabv3_pascal_train_aug/model.ckpt" \
	--vis_logdir="$root_dir/cache_data" \
	--dataset_dir="$root_dir/datasets/PascalContext/tfrecord"
