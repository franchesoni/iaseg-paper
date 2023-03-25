LABEL=4
echo "Label $LABEL"
rm -rf data/perlabel_sbd/annotations
# run the robot, save the terminal output to robot.log and the process number for later
python -u annotate_data.py $LABEL 2>&1 | tee robot.log & robot_pid=$!;
echo "started robot ${robot_pid}"
# run the training loop, save the terminal output to loop.log and the process number for later
python -u continuously_train.py $LABEL 2>&1 | tee loop.log & loop_pid=$!;
echo "started loop ${loop_pid}"
wait $robot_pid
echo "waited ${robot_pid}"
kill $loop_pid
echo "killed ${loop_pid}"
