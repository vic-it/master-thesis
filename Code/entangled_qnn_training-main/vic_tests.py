import time
from datetime import datetime
from timeit import default_timer as timer
from qnns.cuda_qnn import UnitaryParametrization
#from victor_thesis_experiments import *
from victor_thesis_utils import *
from victor_thesis_landscapes import *
from victor_thesis_plots import *
from victor_thesis_metrics import *
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import os

# full experiment framework

#todo
def process_and_store_metrics(landscapes, conf_id, experiment_id):
    print("processing")
    os.makedirs(f"experimental_results/results/runs_{experiment_id}",exist_ok=True)
    file = open(f"experimental_results/results/runs_{experiment_id}/conf_{conf_id}.txt", "w")
    file.write(f"conf_id={conf_id}\n---\n")
    file.close()
    #prep data
    TV_arr = []
    FD_arr = []
    IGSD_arr = []
    SC_arr = []
    #calculate metrics
    for landscape in landscapes:
        TV_arr.append(calc_total_variation(landscape))
        FD_arr.append(calc_fourier_density(landscape))
        IGSD_arr.append(calc_IGSD(landscape))
        SC_arr.append(calc_scalar_curvature(landscape))

    #calculate and store individual sub-metric (avg, std,..)        
    file = open(f"experimental_results/results/runs_{experiment_id}/conf_{conf_id}.txt", "a")
    for idx in range(len(landscapes)):        
        file.write(f"run_{idx}\n")
        file.write(f"TV={TV_arr[idx]}\n")
        file.write(f"FD={FD_arr[idx]}\n")
        file.write(f"IGSD={IGSD_arr[idx]}\n")
        #calculate SC sub-metrics
        #flatten SC
        sc = np.array(SC_arr[idx]).reshape(-1)
        sc_avg = np.mean(sc)
        sc_std = np.std(sc)
        sc_pos= (1.0*np.sum(sc >= 0))/(1.0*len(sc))
        sc_neg = (1.0*np.sum(sc < 0))/(1.0*len(sc))
        sc_abs = np.abs(sc)
        sc_avg_abs = np.mean(sc_abs)
        sc_std_abs = np.std(sc_abs)
        file.write(f"SC_pos={sc_pos}\n")
        file.write(f"SC_neg={sc_neg}\n")
        file.write(f"SC_avg={sc_avg}\n")
        file.write(f"SC_std={sc_std}\n")
        file.write(f"SC_avg_abs={sc_avg_abs}\n")
        file.write(f"SC_std_abs={sc_std_abs}\n---\n")
    # now for all runs combined
    tv = np.array(TV_arr)
    fd =np.array(FD_arr)
    igsd =np.array(IGSD_arr).reshape(-1)
    sc =np.array(SC_arr).reshape(-1)
    sc_pos = (1.0*np.sum(sc >= 0))/(1.0*len(sc))
    sc_neg = (1.0*np.sum(sc < 0))/(1.0*len(sc))
    sc_abs = np.abs(sc)
    file.write("combined\n")
    file.write(f"TV_avg={np.mean(tv)}\n")
    file.write(f"TV_std={np.std(tv)}\n")
    file.write(f"FD_avg={np.mean(fd)}\n")
    file.write(f"FD_std={np.std(fd)}\n")
    file.write(f"IGSD_avg={np.mean(igsd)}\n")
    file.write(f"IGSD_std={np.std(igsd)}\n")
    file.write(f"SC_pos_avg={sc_pos}\n")
    file.write(f"SC_neg_avg={sc_neg}\n")
    file.write(f"SC_avg={np.mean(sc)}\n")
    file.write(f"SC_std={np.std(sc)}\n")
    file.write(f"SC_avg_abs={np.mean(sc_abs)}\n")
    file.write(f"SC_std_abs={np.std(sc_abs)}")
    file.close()

#todo
def store_configs_to_file(unitaries, configurations, experiment_id):
    # config: [type_of_data][num_data_points][deg_of_entanglement][id_unitary][id_try]
    data_type_labels = ["random", "orthogonal", "non_lin_ind","var_s_rank"]
    conf_id = 0    
    file = open(f"experimental_results/configs/configurations_{experiment_id}.txt","w")
    file.write("")
    file.close()
    for data_type in range(len(configurations)):
        for num_data_points in range(len(configurations[data_type])):
            for s_rank in range(len(configurations[data_type][num_data_points])):
                for unitary_id in range(len(configurations[data_type][num_data_points][s_rank])):                   
                    out_string = f"conf_id={conf_id}\n"
                    out_string += f"data_type={data_type_labels[data_type]}\n"
                    out_string += f"num_data_points={num_data_points+1}\n"
                    out_string += f"s_rank={s_rank+1}\n"
                    unitary_string = np.array2string(unitaries[unitary_id].numpy(), separator=',').replace('\n', '').replace(' ', '')
                    out_string += f"unitary={unitary_string}\n"
                    for data_set_id in range(len(configurations[data_type][num_data_points][s_rank][unitary_id])):
                        data_batch = configurations[data_type][num_data_points][s_rank][unitary_id][data_set_id]
                        data_batch_string = np.array2string(data_batch.numpy(),separator=',').replace('\n', '').replace(' ', '')
                        out_string += f"data_batch_{data_set_id}={data_batch_string}\n"
                    out_string +="---\n"
                    file = open(f"experimental_results/configs/configurations_{experiment_id}.txt","a")
                    file.write(out_string)
                    file.close()
                    conf_id += 1

#todo
def generate_data_points(type_of_data, entanglement, num_data_points):
    return generate_random_datapoints(1, 1, random_unitary_matrix(1))

# one asynchronous run will calculate 5 landscapes and their metrics
def run_single_experiment_batch(grid_size, dimensions, data_batch, U, qnn, conf_id, experiment_id):
    #data batch contains 5 datapoint-sets, as we do 5 runs per unitary and then average etc.
    landscapes= []
    for data_set in data_batch:
        landscapes.append(generate_loss_landscape(grid_size, dimensions, data_set, U, qnn))
    process_and_store_metrics(landscapes, conf_id, experiment_id)
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"[{now}] Finished run: {conf_id}")

def run_full_experiment():
    num_unitaries = 5
    num_tries = 5
    num_qubits = 1
    grid_size = 3
    dimensions = 3
    # generate an experiment id (based on time)
    current_time = datetime.now()
    exp_id = str(grid_size)+"_"+str(dimensions)+"_"+str(current_time.month)+"_"+str(current_time.day)+"_"+str(current_time.hour)+"_"+str(current_time.minute)+"_"+str(current_time.second)
    # create directories for results and configs
    
    os.makedirs("experimental_results/configs",exist_ok=True)
    os.makedirs("experimental_results/results",exist_ok=True)
    # generate general qnn (?)
    qnn = UnitaryParametrization(num_wires=1, num_layers=2, device='cpu')

    unitaries = []
    # [type_of_data][num_data_points][deg_of_entanglement][id_unitary][id_try]
    configurations = []
    for _ in range(num_unitaries):
        #generate a random unitary with num_qubits qubits (why are they the same?)
        unitaries.append(torch.tensor(np.array(random_unitary_matrix(num_qubits)), dtype=torch.complex128, device="cpu"))

    
    start = time.time()
    # generate configurations (5 datapoint sets = 5 runs per config)
    conf_id = 0    
    #cpu_count()
    with ProcessPoolExecutor(cpu_count()) as exe:        
        # iterate over  type of training data: 0=random, 1=orthogonal, 2=linearly dependent in H_x, 3= variable schmidt rank
        for type_of_data in range(1,5,1):   
            num_data_points_row = []
            # iterate over training data size 1 to 4
            for num_data_points in range(1,5,1):
                deg_of_entanglement_row = []
                # iterate over degree of entanglement 1 to 4
                for deg_of_entanglement in range(1,5,1):
                    # iterate over unitaries
                    unitary_row = []
                    for unitary in unitaries:       
                        data_batch_for_unitary = []
                        #iterate over number of tries/runs
                        for _ in range(1,num_tries+1,1):
                            # generate array of training data configurations [type_of_data][num_data_points][deg_of_entanglement][id_unitary][id_try]
                            data_points = generate_data_points(type_of_data, deg_of_entanglement, num_data_points)
                            data_batch_for_unitary.append(data_points)
                        # run this per configuration unitary (5 sets of data -> take average and stdv...)
                        #exe.submit(run_single_experiment_batch,grid_size, dimensions, data_batch_for_unitary, unitary, qnn, conf_id, exp_id)
                        run_single_experiment_batch(grid_size, dimensions, data_batch_for_unitary, unitary, qnn, conf_id, exp_id)
                        conf_id += 1
                        unitary_row.append(data_batch_for_unitary)                    
                    deg_of_entanglement_row.append(unitary_row)
                num_data_points_row.append(deg_of_entanglement_row)
            configurations.append(num_data_points_row)

    store_configs_to_file(unitaries, configurations, exp_id)    
    end = time.time()
    print(f"total runtime: {np.round(end-start,2)}s")
if __name__ == '__main__':
    # one thread per core
    torch.set_num_threads(1)
    torch.multiprocessing.set_sharing_strategy('file_system')   
    run_full_experiment()