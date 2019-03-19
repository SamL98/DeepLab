name="${1}"
data_dir="calib_data_${name}"
slices_name="slices_${name}.pkl"

python do_inf.py --name=$name --slice_file=$slices_name 
