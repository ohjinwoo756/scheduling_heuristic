### USER DEFINED ###
# Set processors used in scheduling
processor = ['cpu', 'gpu', 'npu']
# processor = ['cpu', 'gpu']

input_dir_path = "galaxyS9_affin_profile_final"
# input_dir_path = "hikey970_profile"

est_type = ["npu_x3", "basic", "npu_x10", "npu_x15"]

# Feasible values on command args
schedulers = ["GA", "ManualMapping", "AppBasedHeuristic", "LayerBasedHeuristic", "JHeuristic"]
CPU_intraParall = ['4', '31', '22', '211', '1111']
app_to_obj_dict = ['Throughput', 'Response_time']
app_to_cst_dict = ['Deadline', 'None']
CPU_util = ['100', '70', '60', '50', '40']

# Used
networks = []
cfg_prototxt_path = []
est_prototxt_path = []
start_nodes_idx = []
end_nodes_idx = []
