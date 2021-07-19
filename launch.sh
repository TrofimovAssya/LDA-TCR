k=$1
st=$2
python lda_kmers.py --k $k --seq_type $st --dataset_name ${st}_${k}mers.csv

