python main.py --model er --dataset cifar10 --n_classes 10 --n_tasks 5 --buffer_size 200 --lr 0.03 --n_epochs 20 --device_id 0

python main.py --model er --dataset cifar100 --n_classes 100 --n_tasks 20 --buffer_size 200 --lr 0.03 --n_epochs 50 --device_id 0

python main.py --model er --dataset perm-mnist --n_classes 10 --n_tasks 20 --buffer_size 200 --lr 0.1 --n_epochs 1 --batch_size 128 --device_id 0

python main.py --model er --dataset rot-mnist --n_classes 10 --n_tasks 20 --buffer_size 200 --lr 0.1 --n_epochs 1 --batch_size 128 --device_id 0
