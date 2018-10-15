root_dir=/Users/samlerner/projects/deeplab
python ~/models/research/deeplab/train.py \
	--logtostderr \
	--training_number_of_steps=30000 \
	--train_split="train" \
	--model_variant="resnet_v1_50_beta" \
	--atrous_rates=6 \
	--atrous_rates=12 \
	--atrous_rates=18 \
	--output_stride=16 \
	--decoder_output_stride=4 \
	--train_crop_size=400 \
	--train_crop_size=400 \
	--train_batch_size=1 \
	--dataset="pascal_voc_seg" \
	--tf_initial_checkpoint="$root_dir/model_trained/resnet_v1_50/model.ckpt" \
	--train_logdir="$root_dir/cache_data" \
	--dataset_dir="$root_dir/datasets/PascalContext/tfrecord"
