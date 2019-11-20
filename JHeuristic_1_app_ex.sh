


heuristic='JHeuristic'
networks=('squeezenet' 'mobilenet_v1' 'mobilenet_v2' 'densenet40_32')
objective='Response_time'
pe_config=('cpu' 'cpu gpu' 'cpu gpu npu')
pe_config_c=('c' 'cg' 'cgn')
cpu_config=('4' '22' '31' '211' '1111')

for ((n1_idx=0; n1_idx < ${#networks[@]}; n1_idx++)); do # for all networks
	for ((p_idx=0; p_idx < ${#pe_config[@]}; p_idx++)); do # for all pe  configuration
		for ((c_idx=0; c_idx < ${#cpu_config[@]}; c_idx++)); do # for all cpu configuration

			if [ ${p_idx} -eq 0 -a ${c_idx} -eq 0 ]; then # except for '4' when 'cpu'
				continue
			fi

			echo ${networks[n1_idx]} ${pe_config[p_idx]} ${cpu_config[c_idx]}
			echo "==========================================================="
			python src/main.py -s ${heuristic} -n ${networks[n1_idx]} -o ${objective} -r ${pe_config[p_idx]} -c ${cpu_config[c_idx]} > ${heuristic}_${networks[n1_idx]}_${pe_config_c[p_idx]}_${cpu_config[c_idx]}.log

		done
	done
done



