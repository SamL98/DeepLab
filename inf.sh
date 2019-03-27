name="${1}"
data_dir="calib_data/${name}"
slices_name="calib_data/${name}/slices.pkl"

python do_inf.py --name=$name --slice_file=$slices_name 
