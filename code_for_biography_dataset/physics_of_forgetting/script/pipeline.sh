export PYTHONPATH=./
python ./training/pre_training.py

export PYTHONPATH=./
python ./training/full_parameter_fine_tuning.py --learning_rate 1e-3 --weight_decay 1e-2
export PYTHONPATH=./
python ./training/full_parameter_fine_tuning.py --learning_rate 3e-4 --weight_decay 1e-3
export PYTHONPATH=./
python ./training/full_parameter_fine_tuning.py --learning_rate 1e-4 --weight_decay 1e-2
export PYTHONPATH=./
python ./training/full_parameter_fine_tuning.py --learning_rate 1e-3 --weight_decay 1e-3
export PYTHONPATH=./
python ./training/full_parameter_fine_tuning.py --learning_rate 3e-4 --weight_decay 1e-2
export PYTHONPATH=./
python ./training/full_parameter_fine_tuning.py --learning_rate 1e-4 --weight_decay 1e-3