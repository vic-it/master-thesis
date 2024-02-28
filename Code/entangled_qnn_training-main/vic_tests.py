import time
from datetime import datetime
from timeit import default_timer as timer
from victor_thesis_experiments import *
from victor_thesis_utils import *
from victor_thesis_landscapes import *
from victor_thesis_plots import *
from victor_thesis_metrics import *
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

# full experiment framework

#todo
def process_and_store_metrics(landscapes, conf_id):
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
def store_configs_to_file(unitaries, configurations):
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
    return type_of_data*entanglement*num_data_points

# one asynchronous run will calculate 5 landscapes and their metrics
def run_single_experiment(grid_size, dimensions, data_batch, U, qnn, conf_id):
    #data batch contains 5 datapoint-sets, as we do 5 runs per unitary and then average etc.
    landscapes= []
    for data_set in data_batch:
        landscapes.append(generate_loss_landscape(grid_size, dimensions, data_set, U, qnn))
    process_and_store_metrics(landscapes, conf_id)
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"[{now}] Finished run: {conf_id}")

def run_full_experiment(num_qubits, num_unitaries = 5, num_tries = 5):
    # generate general qnn (?)
    qnn = UnitaryParametrization(num_wires=num_qubits, num_layers=1, device='cpu')
    grid_size = 2
    dimensions = 2
    unitaries = []
    #[id_unitary][id_try][type_of_data][deg_of_entanglement][num_data_points]
    configurations = []
    for _ in range(num_unitaries):
        #generate a random unitary with num_qubits qubits (why are they the same?)
        unitaries.append(torch.tensor(np.array(random_unitary_matrix(num_qubits)), dtype=torch.complex128, device="cpu"))

    
    start = time.time()
    # generate configurations (5 datapoint sets = 5 runs per config)
    conf_id = 0    
    with ProcessPoolExecutor(cpu_count()) as exe:        
        # iterate over  type of training data: 1=random, 2=orthogonal, 3=linearly dependent in H_x, 4= variable schmidt rank
        for type_of_data in range(1,5,1):   
            num_data_points_row = []
            # iterate over degree of entanglement 1 to 4
            for num_data_points in range(1,5,1):
                deg_of_entanglement_row = []
                # iterate over training data size 1 to 4
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
                            # run experiment on individual core
                        # run this per configuration X unitary (5 sets of data -> take average and stdv...)
                        exe.submit(run_single_experiment,grid_size, dimensions, data_batch_for_unitary, unitary, qnn, conf_id)                 
                        conf_id += 1
                        unitary_row.append(data_batch_for_unitary)                    
                    deg_of_entanglement_row.append(unitary_row)
                num_data_points_row.append(deg_of_entanglement_row)
            configurations.append(num_data_points_row)

    configurations = np.array(configurations)
    print(configurations.shape)  
    store_configs_to_file(unitaries, configurations)    
    end = time.time()
    print(f"total runtime: {np.round(end-start,2)}s")
if __name__ == '__main__':
    run_full_experiment(2)