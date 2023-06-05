modified from pytorch-a2c-ppo-acktr

## example run
```bash
python3 main.py --cuda_id 3 --env_sub_name identical_20_vs_20 --batch_size 3000 --num-steps 8  --seed 5 
python3 main.py --cuda_id 1 --env_sub_name identical_10_vs_10 --batch_size 1000 --num-steps 8  --seed 5  
python3 main.py --cuda_id 2 --env_sub_name identical_5_vs_5 --batch_size 1000 --num-steps 8  --seed 5  
python3 main.py --cuda_id 7 --env_sub_name identical_2_vs_2 --batch_size 1000 --num-steps 8 --seed 5  
python3 main.py --cuda_id 4 --env_sub_name identical_20_vs_5 --batch_size 3000 --num-steps 8  --seed 5  
python3 main.py --cuda_id 7 --env_sub_name identical_5_vs_1 --batch_size 1000 --num-steps 8 --seed 5   
python3 main.py --cuda_id 7 --env_sub_name special_20_vs_20 --batch_size 3000 --num-steps 8 --seed 5   
python3 main.py --cuda_id 5 --env_sub_name special_10_vs_10 --batch_size 1000 --num-steps 8 --seed 5   
python3 main.py --cuda_id 0 --env_sub_name special_5_vs_5 --batch_size 1000 --num-steps 8 --seed 5  
python3 main.py --cuda_id 1 --env_sub_name special_20_vs_5 --batch_size 3000 --num-steps 8 --seed 5 
python3 main.py --cuda_id 2 --env_sub_name special_5_vs_1 --batch_size 1000 --num-steps 8 --seed 5  
```