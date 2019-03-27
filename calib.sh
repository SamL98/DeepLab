name="${1}"
data_dir="calib_data/${name}"
slices_name="calib_data/${name}/slices.pkl"

python do_calib.py --sm_by_slice --output_file=$slices_name --data_dir=$data_dir --sigma=0.015
