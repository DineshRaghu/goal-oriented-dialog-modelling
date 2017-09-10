#echo "1"
#python single_dialog.py --train False --task_id 1 --embedding_size 128 --hops 1 --interactive False --OOV False --model_dir model_baseline/ --logs_dir logs_baseline/ >> logs_baseline/test_accuracies.log 2>> logs_baseline/single_dialog.err
#python single_dialog.py --train False --task_id 1 --embedding_size 128 --hops 1 --interactive False --OOV True --model_dir model_baseline/ --logs_dir logs_baseline/ >> logs_baseline/test_accuracies.log 2>> logs_baseline/single_dialog.err

echo "2"
python single_dialog.py --train False --task_id 2 --embedding_size 32 --hops 1 --interactive False --OOV False --model_dir model_baseline/ --logs_dir logs_baseline/ >> logs_baseline/test_accuracies.log 2>> logs_baseline/single_dialog.err
python single_dialog.py --train False --task_id 2 --embedding_size 32 --hops 1 --interactive False --OOV True --model_dir model_baseline/ --logs_dir logs_baseline/ >> logs_baseline/test_accuracies.log 2>> logs_baseline/single_dialog.err

echo "3"
python single_dialog.py --train False --task_id 3 --embedding_size 32 --hops 3 --interactive False --OOV False --model_dir model_baseline/ --logs_dir logs_baseline/ >> logs_baseline/test_accuracies.log 2>> logs_baseline/single_dialog.err
python single_dialog.py --train False --task_id 3 --embedding_size 32 --hops 3 --interactive False --OOV True --model_dir model_baseline/ --logs_dir logs_baseline/ >> logs_baseline/test_accuracies.log 2>> logs_baseline/single_dialog.err

echo "4"
python single_dialog.py --train False --task_id 4 --embedding_size 128 --hops 2 --interactive False --OOV False --model_dir model_baseline/ --logs_dir logs_baseline/ >> logs_baseline/test_accuracies.log 2>> logs_baseline/single_dialog.err
python single_dialog.py --train False --task_id 4 --embedding_size 128 --hops 2 --interactive False --OOV True --model_dir model_baseline/ --logs_dir logs_baseline/ >> logs_baseline/test_accuracies.log 2>> logs_baseline/single_dialog.err

echo "5"
python single_dialog.py --train False --task_id 5 --embedding_size 32 --hops 3 --interactive False --OOV False --model_dir model_baseline/ --logs_dir logs_baseline/ >> logs_baseline/test_accuracies.log 2>> logs_baseline/single_dialog.err
python single_dialog.py --train False --task_id 5 --embedding_size 32 --hops 3 --interactive False --OOV True --model_dir model_baseline/ --logs_dir logs_baseline/ >> logs_baseline/test_accuracies.log 2>> logs_baseline/single_dialog.err

#echo "6"
#python single_dialog.py --train False --task_id 6 --embedding_size 128 --hops 4 --interactive False --OOV False --model_dir model_baseline/ --logs_dir logs_baseline/ >> logs_baseline/test_accuracies.log 2>> logs_baseline/single_dialog.err