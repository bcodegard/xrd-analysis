
for ((i=$1;i<=$2;i++)) do

	# todo: 
	# skip if the .ant file exists locally already
	# unless a flag is given to re-download

	# check if file exists on remote
	if curl --output /dev/null --silent --head --fail "http://cms2.physics.ucsb.edu/DRS/Run${i}/Run${i}_pulses.ant"
	then
		echo "run ${i} exists; downloading it"
		curl "http://cms2.physics.ucsb.edu/DRS/Run${i}/Run${i}_pulses.ant" --output "./data/rpi/ant/Run${i}_pulses.ant"
	else
		echo "run ${I} does not exist"
	fi

done;
