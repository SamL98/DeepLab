name="${1}"
data_dir="calib_data/${name}"
slices_name="calib_data/${name}/slices.pkl"

python do_calib.py --sigma=0.02 --output_file=$slices_name --data_dir=$data_dir 
