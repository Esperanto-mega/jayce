this=jayce
data_root=~/$this/Data/BA3
model_path=~/$this/Model/ba3net.pt
result_path=~/$this/result

python main.py \
    --data_root $data_root \
    --model_path $model_path \
    --result $result_path \
    --cuda 7
