name="${1}"
data_dir="calib_data/${name}"
slices_name="calib_data/${name}/slices.pkl"

if [[ -z $DS_PATH ]]; then
	if [[ -z DATAPATH ]]; then
		ds_path="D:/datasets/processed/voc2012"
	else
		ds_path=$DATAPATH
	fi
else
	ds_path=$DS_PATH
fi

if [[ -z $2 ]]; then
	output_name=$name
else
	output_name="${2}"
fi

mkdir "${ds_path}/deeplab_prediction/test/${output_name}"
python do_inf.py --name=$name --output_name=$output_name --slice_file=$slices_name --alpha=0.8 
