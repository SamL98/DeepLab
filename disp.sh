name="${1}"
slices_name="slices_${name}.pkl"
idx=-1

if [ $# -eq 2 ]
then
	idx=$2
fi

python do_disp.py --idx=$idx --name=$name --slice_file=$slices_name 
