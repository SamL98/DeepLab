name="${1}"
data_dir="calib_data_${name}"
slices_name="slices_${name}.pkl"

pred_dir="/cygdrive/d/datasets/processed/voc2012/deeplab_prediction/test/${name}"
rm -rf pred_dir
mkdir pred_dir
python do_inf.py --name=$name --slice_file=$slices_name 
