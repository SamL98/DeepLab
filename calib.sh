name="${1}"
data_dir="calib_data_${name}"
slices_name="slices_${name}.pkl"

python do_calib.py --output_file=$slices_name --data_dir=$data_dir --sigma=0.015
