root_dir=/Users/samlerner/projects/deeplab
python ~/models/research/deeplab/train.py \
	--logtostderr \
	--training_number_of_steps=20000 \
	--train_split="train" \
	--model_variant="xception_65" \
	--atrous_rates=6 \
	--atrous_rates=12 \
	--atrous_rates=18 \
	--output_stride=16 \
	--decoder_output_stride=4 \
	--train_crop_size=513 \
	--train_crop_size=513 \
	--train_batch_size=1 \
	--fine_tune_batch_norm=true \
	--dataset="pascal_voc_seg" \
	--tf_initial_checkpoint="$root_dir/model_trained/deeplabv3_pascal_train_aug/model.ckpt" \
	--train_logdir="$root_dir/cache_data/model-$(date +%s)/train" \
	--dataset_dir="$root_dir/datasets/voc2012"
