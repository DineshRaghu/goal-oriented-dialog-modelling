declare -a arr=("1" "2" "3" "4" "5")
for ID in "${arr[@]}"
do
	echo $ID
	python single_dialog.py --train False --task_id $ID --interactive False --OOV False >> test_accuracies.log 2>> delete_this.err
	python single_dialog.py --train False --task_id $ID --interactive False --OOV True >> test_accuracies.log 2>> delete_this.err
done
echo "6"
python single_dialog.py --train False --task_id 6 --interactive False --OOV False >> test_accuracies.log 2>> delete_this.err