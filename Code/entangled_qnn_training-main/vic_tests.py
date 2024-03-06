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
    #calculates and stores the raw metrics, the standard deviations and medians of the metrics and have config id in name of file to match to configs.txt
    #gets 5 landscapes as input
    #conf_id
    #run_1
    #TV=...
    #...
    #run_5
    #...
    #run_avg
    #TV=...
    #...
    #run_stdv
    #TV=...
    #...
    return 0

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
                    out_string += f"num_data_points={num_data_points}\n"
                    out_string += f"s_rank={s_rank}\n"
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
        
    #write down all run configurations in a file with the run_ids
    #-
    #conf_id=16
    #data_type=orthogonal
    #num_data_points=3
    #deg_entanglement=2
    #unitary=[[a,b],[c,d]] (rough form)
    #data_batch_1=[[...]]
    #...
    #data_batch_5=[[...]]
    return 0

#todo
def generate_data_points(type_of_data, entanglement, num_data_points):
    return generate_random_datapoints(3, 1, random_unitary_matrix(1))

# one asynchronous run will calculate 5 landscapes and their metrics
def run_single_experiment(grid_size, dimensions, data_batch, U, qnn, conf_id, experiment_id):
    #data batch contains 5 datapoint-sets, as we do 5 runs per unitary and then average etc.
    landscapes= []
    # for data_set in data_batch:
    #     landscapes.append(generate_loss_landscape(grid_size, dimensions, data_set, U, qnn))
    # process_and_store_metrics(landscapes, conf_id, experiment_id)
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"[{now}] Finished run: {conf_id}")

def run_full_experiment(num_qubits, num_unitaries = 5, num_tries = 5):
    # generate an experiment id (based on time)
    current_time = datetime.now()
    exp_id = ""+str(current_time.month)+"_"+str(current_time.day)+"_"+str(current_time.hour)+"_"+str(current_time.minute)+"_"+str(current_time.second)
    # create directories for results and configs
    
    os.makedirs("experimental_results/configs",exist_ok=True)
    os.makedirs("experimental_results/results",exist_ok=True)
    # generate general qnn (?)
    qnn = UnitaryParametrization(num_wires=num_qubits, num_layers=1, device='cpu')
    grid_size = 2
    dimensions = 2
    unitaries = []
    # [type_of_data][num_data_points][deg_of_entanglement][id_unitary][id_try]
    configurations = []
    for _ in range(num_unitaries):
        #generate a random unitary with num_qubits qubits (why are they the same?)
        unitaries.append(torch.tensor(np.array(random_unitary_matrix(num_qubits)), dtype=torch.complex128, device="cpu"))

    
    start = time.time()
    # generate configurations (5 datapoint sets = 5 runs per config)
    conf_id = 0    
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
                        exe.submit(run_single_experiment,grid_size, dimensions, data_batch_for_unitary, unitary, qnn, conf_id, exp_id)                 
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
    run_full_experiment(2)