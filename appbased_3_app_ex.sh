


heuristic='AppBasedHeuristic'
networks=('squeezenet' 'mobilenet_v1' 'mobilenet_v2' 'densenet40_32')
objective='Response_time'
constraint='Deadline'
pe_config=('cpu' 'cpu gpu' 'cpu gpu npu')
pe_config_c=('c' 'cg' 'cgn')
cpu_config=('4' '22' '31' '211' '1111')

elapsed_sq_c=(53860	45971 56835	66815)
elapsed_mnv1_c=(90827 85700 90827 129462)
elapsed_mnv2_c=(157921 117825 157398 181016)
elapsed_dn_c=(186472 148701 186384 244897)
elapsed_all_c=(
	elapsed_sq_c[@]
	elapsed_mnv1_c[@]
	elapsed_mnv2_c[@]
	elapsed_dn_c[@]
)

elapsed_sq_cg=(27525 27525 27525 27525 27525) 
elapsed_mnv1_cg=(30529 30563 30541 30563 30613)
elapsed_mnv2_cg=(48316 48460 48301 48460 48453)
elapsed_dn_cg=(54716 54716 54716 54716 54716)
elapsed_all_cg=(
	elapsed_sq_cg[@]
	elapsed_mnv1_cg[@]
	elapsed_mnv2_cg[@]
	elapsed_dn_cg[@]
)

elapsed_sq_cgn=(10293 10293 10293 10293 10293) 
elapsed_mnv1_cgn=(13666 13666 13666 13666 13666)
elapsed_mnv2_cgn=(16942 16942 16942 16942 16942)
elapsed_dn_cgn=(18707 18707 18707 18707 18707)
elapsed_all_cgn=(
	elapsed_sq_cgn[@]
	elapsed_mnv1_cgn[@]
	elapsed_mnv2_cgn[@]
	elapsed_dn_cgn[@]
)

progress=0
for ((n1_idx=0; n1_idx < ${#networks[@]}; n1_idx++)); do # for priority 1 
	for ((n2_idx=0; n2_idx < ${#networks[@]}; n2_idx++)); do # for priority 2
		if [ ${n2_idx} -le ${n1_idx} ]; then # app selection
			continue
		fi

		for ((n3_idx=0; n3_idx < ${#networks[@]}; n3_idx++)); do # for priority 3
			if [ ${n3_idx} -le ${n2_idx} ]; then # app selection
				continue
			fi

			for ((p_idx=0; p_idx < ${#pe_config[@]}; p_idx++)); do # for all pe  configuration
				for ((c_idx=0; c_idx < ${#cpu_config[@]}; c_idx++)); do # for all cpu configuration

					if [ ${p_idx} -eq 0 -a ${c_idx} -eq 0 ]; then # except for '4' when 'cpu'
						continue
					fi

					if [ ${p_idx} -eq 0 ]; then # if CPU
						elapsed_of_n1=${!elapsed_all_c[n1_idx]:$c_idx-1:1}
						elapsed_of_n2=${!elapsed_all_c[n2_idx]:$c_idx-1:1}
						elapsed_of_n3=${!elapsed_all_c[n3_idx]:$c_idx-1:1}
						deadline=`expr $elapsed_of_n1 + $elapsed_of_n2 + $elapsed_of_n3 + 20000`
					elif [ ${p_idx} -eq 1 ]; then # if CPU, GPU
						elapsed_of_n1=${!elapsed_all_cg[n1_idx]:$c_idx:1}
						elapsed_of_n2=${!elapsed_all_cg[n2_idx]:$c_idx:1}
						elapsed_of_n3=${!elapsed_all_cg[n3_idx]:$c_idx:1}
						deadline=`expr $elapsed_of_n1 + $elapsed_of_n2 + $elapsed_of_n3 + 20000`
					else # if CPU, GPU, NPU
						elapsed_of_n1=${!elapsed_all_cgn[n1_idx]:$c_idx:1}
						elapsed_of_n2=${!elapsed_all_cgn[n2_idx]:$c_idx:1}
						elapsed_of_n3=${!elapsed_all_cgn[n3_idx]:$c_idx:1}
						deadline=`expr $elapsed_of_n1 + $elapsed_of_n2 + $elapsed_of_n3 + 20000`
					fi

					echo ${networks[n1_idx]} ${networks[n2_idx]} ${networks[n3_idx]} 
					echo ${pe_config[p_idx]} ${cpu_config[c_idx]}
					echo ${elapsed_of_n1} ${elapsed_of_n2} ${elapsed_of_n3} ${deadline}
					echo "==============================================================================="
					python src/main.py -s ${heuristic} -n ${networks[n1_idx]} ${networks[n2_idx]} ${networks[n3_idx]} -o ${objective} ${objective} ${objective} -t ${constraint} ${constraint} ${constraint} -d ${deadline} ${deadline} ${deadline} -r ${pe_config[p_idx]} -c ${cpu_config[c_idx]} > ${heuristic}_${networks[n1_idx]}_${networks[n2_idx]}_${networks[n3_idx]}_${deadline}_${deadline}_${deadline}_${pe_config_c[p_idx]}_${cpu_config[c_idx]}.log &

					progress=`expr $progress + 1`
					if [ `expr $progress % 8` == 0 ]; then
						wait
					fi

				done
			done
		done
	done
done
wait



