jbsub -queue x86_7d -mem 4g -out task3.train.weighted.out -err task3.train.weighted.err /u/diraghu1/miniconda3/bin/python single_dialog.py --train True --task_id 3 --embedding_size 32 --hops 3 --interactive False --OOV False --model_dir model_baseline/ --epochs 2000 --loss_type weighted
