

python main.py --dataset=cora  --total_epochs=400  --log log_cora.txt  --Q_training  --model_path cora_GCN --weight 0.4

python main.py --dataset=citeseer  --total_epochs=400 --log log_citeseer.txt  --Q_training --model_path citeseer_GCN --weight 0.1

python main.py --dataset=pubmed  --total_epochs=400  --log log_pubmed.txt --Q_training --model_path pubmed_GCN --weight 0.9  #i_episode 16

python main.py --dataset=ms_academic_cs  --total_epochs=400  --log log_cs.txt --Q_training  --model_path ms_academic_cs_GCN --weight 0.1

python main.py --dataset=amazon_electronics_computers  --total_epochs=400  --log log_amazon_computer.txt  --Q_training  --model_path amazon_electronics_computers_GCN2 --weight 0.3

python main.py --dataset=amazon_electronics_photo  --total_epochs=400   --log log_amazon_photo.txt  --Q_training  --model_path amazon_electronics_photo_GCN --weight 0.1



seeds='40 41 42 43 44 45 46 47 48 49'
for seed in $seeds
do 
    python train.py --seed $seed  --dataset cora
done

