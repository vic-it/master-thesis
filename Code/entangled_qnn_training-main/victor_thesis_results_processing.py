from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from victor_thesis_plots import plot_results_metric
from victor_thesis_utils import calc_average_of_std
class Result:     
    def __init__(self, idx):
        self.idx = idx
        self.TV_list = []
        self.FD_list=[]
        self.IGSD_list=[]
        self.SC_pos_list=[]
        self.SC_neg_list=[]
        self.SC_avg_list=[]
        self.SC_std_list=[]
        self.SC_avg_abs_list=[]
        self.SC_std_abs_list=[]
class Combined_Result:
    def __init__(self):
        self.config_indices = []
        self.number_configs = 0
        self.number_runs = 0
        self.TV_avg= 0
        self.TV_std= 0
        self.FD_avg= 0
        self.FD_std= 0
        self.IGSD_avg= 0
        self.IGSD_std= 0
        self.SC_pos_avg= 0
        self.SC_neg_avg= 0
        self.SC_avg= 0
        self.SC_std= 0
        self.SC_abs_avg= 0
        self.SC_abs_std= 0

def read_results(idx, max_run_id):
    results = []
    path = f"experimental_results/results/runs_{idx}/"
    for target_idx in range(max_run_id):
        current_result = Result(target_idx)
        for u_offset in range(5):
            file_idx =5*target_idx+u_offset
            file =f"{path}conf_{file_idx}.txt"
            f = open(file, "r")
            for line in f:
                split_line = line.split("=")
                if len(split_line)>1:
                    split_line[1].replace("\n","")
                    if split_line[0]=="IGSD":
                        igsds = split_line[1].replace("[","").replace("]","").split(",")
                        for igsd in igsds:
                            current_result.IGSD_list.append(float(igsd))
                    else:
                        value = float(split_line[1])
                        metric = split_line[0]                        
                        if metric == "conf_id":
                                continue
                        else:
                            getattr(current_result, f"{metric}_list").append(value)
                if(split_line[0]=="combined\n"):
                    break
        results.append(current_result)
    return results

def generate_config():
    config_by_id = []
    for type_of_data in range(1, 5, 1):
            # iterate over training data size 1 to 4
            for num_data_points in range(1, 5, 1):
                # iterate over degree of entanglement 1 to 4
                for deg_of_entanglement in range(1, 5, 1):
                    # iterate over number of tries/runs               
                    config = []
                    config.append(type_of_data)
                    config.append(num_data_points)
                    config.append(deg_of_entanglement)
                    config_by_id.append(config)
    return config_by_id
                        
def get_results_where(results, configs, type_of_data = 0, num_data_points = 0, schmidt_rank = 0):
    output = []
    for idx, config in enumerate(configs):
        if type_of_data == 0 or config[0] == type_of_data:
            if num_data_points == 0 or config[1] == num_data_points:
                if schmidt_rank == 0 or config[2] == schmidt_rank:
                    output.append(results[idx])
    return output

def combine_results(result_list):
    combined_results = Combined_Result()    
    TV_list = []
    FD_list=[]
    IGSD_list=[]
    SC_pos_list=[]
    SC_neg_list=[]
    SC_avg_list=[]
    SC_std_list=[]
    SC_avg_abs_list=[]
    SC_std_abs_list=[]
    #concatenate all lists
    for result in result_list: 
        combined_results.config_indices.append(result.idx) 
        combined_results.number_configs += 1
        combined_results.number_runs += 25      
        TV_list+=result.TV_list
        FD_list+=result.FD_list
        IGSD_list+=result.IGSD_list
        SC_pos_list+=result.SC_pos_list
        SC_neg_list+=result.SC_neg_list
        SC_avg_list+=result.SC_avg_list
        SC_std_list+=result.SC_std_list
        SC_avg_abs_list+=result.SC_avg_abs_list
        SC_std_abs_list+=result.SC_std_abs_list
    # TV metrics
    combined_results.TV_avg=np.mean(TV_list)
    combined_results.TV_std=np.std(TV_list)
    # FD metrics
    combined_results.FD_avg=np.mean(FD_list)
    combined_results.FD_std=np.std(FD_list)
    # IGSD metrics
    combined_results.IGSD_avg=np.mean(IGSD_list)
    combined_results.IGSD_std=np.std(IGSD_list)
    # SC metrics
    combined_results.SC_pos_avg=np.mean(SC_pos_list)
    combined_results.SC_neg_avg=np.mean(SC_neg_list)
    combined_results.SC_avg=np.mean(SC_avg_list)
    combined_results.SC_abs_avg=np.mean(SC_avg_abs_list)
    # SC std stuff
    combined_results.SC_std= calc_average_of_std(SC_std_list)
    combined_results.SC_abs_std=calc_average_of_std(SC_std_abs_list)
    return combined_results

def visualize_metrics(combined_results_list, x_label, title, sample_labels = range(1, 5)):    
    attr_list = ["TV", "IGSD","FD", "SC", "SC_abs"]
    combined_mean_list = []
    combined_std_list =[]
    for attr_name in attr_list:
        combined_results_mean = []
        combined_results_std = []
        for res in combined_results_list:
            combined_results_mean.append(getattr(res,f"{attr_name}_avg"))
            combined_results_std.append(getattr(res,f"{attr_name}_std"))
        combined_mean_list.append(combined_results_mean)
        combined_std_list.append(combined_results_std)
    pos_list = []
    neg_list = []
    for res in combined_results_list:
        pos_list.append(res.SC_pos_avg)
        neg_list.append(res.SC_neg_avg)
    plot_results_metric(combined_mean_list, combined_std_list, pos_list, neg_list, attr_list, x_label, title, sample_labels)
    
def get_results(idx):
    # condensed configs from 1600total runs -> 320 configXunitary -> into 64 configurations, consiting of lists of results
    configs = generate_config()
    #print(configs)
    results = read_results(idx, len(configs))
    return results, configs


run_id = "10_6_3_26_14_44_31"
get_results(run_id)
