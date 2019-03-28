name="${1}"
data_dir="calib_data/${name}"
slices_name="calib_data/${name}/slices.pkl"

ds_path="D:/datasets/processed/voc2012"
if [[ -z $DS_PATH ]]; then
	ds_path=$DS_PATH
fi

mkdir "${ds_path}/deeplab_prediction/test/${name}"
python do_inf.py --name=$name --slice_file=$slices_name 
