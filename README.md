# Databricks_IDS

Intrusion Detection Systems (IDSs) and Intrusion Prevention Systems (IPSs) are the most important defense tools against the sophisticated and ever-growing network attacks. Given network flow data, can we detect different network attacks?

# Network attacks 
There are different types of network attacks such as DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest.

# Features
Total 80 features are taken from network flow data,including Label. 

# ID : 
record id

# Label : 
BENIGN(Normal),DoS(Attack)

# Packet info : 
              'Total Fwd Packets','Total Backward Packets', 'Total Length of Fwd Packets',
  	          'Total Length of Bwd Packets', 'Fwd Packet Length Max',
 	            'Fwd Packet Length Min', 'Fwd Packet Length Mean',
 	            'Fwd Packet Length Std', 'Bwd Packet Length Max',
 	            'Bwd Packet Length Min', 'Bwd Packet Length Mean',
 	            'Bwd Packet Length Std'

# Payload info : 
               'Flow Bytes/s', 'Flow Packets/s',
               'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
               'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
               'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
               'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
               'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
               'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
               'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
               'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
               'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
               'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
               'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
               'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
               'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
               'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
               'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
               'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward'
               
# Session info :
               'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
               'Idle Std', 'Idle Max', 'Idle Min'

# MLFLOW Project
An MLflow Project is a format for packaging data science code in a reusable and reproducible way, based primarily on conventions. In addition, the Projects component includes an API and command-line tools for running projects, making it possible to chain together projects into workflows.

# Running the MLFLOW Project

# 1) Databricks Notebook
```
import mlflow
import warnings
from mlflow import projects

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()<br />
dbutils.fs.put("file:///root/.databrickscfg","[DEFAULT]\nhost=https://community.cloud.databricks.com\ntoken = "+token,overwrite=True)<br />

#Testing Model with Random Parameters
parameters = [{'n_estimators': 10},
              {'n_estimators': 50},
              {'n_estimators': 100}]

#set the project git url
ml_project_uri = "git://github.com/iRahulP/Databricks_IDS.git"

warnings.filterwarnings("ignore", category=DeprecationWarning)
# Iterate over three different runs with different parameters
for param in parameters:
  print(f"Running with param = {param}"),
  res_sub = projects.run(ml_project_uri, parameters=param)
  print(f"status={res_sub.get_status()}")
  print(f"run_id={res_sub.run_id}")
```

```
Run the Notebook
```

# 2) Any Local Environment ~ Linux Terminal | Windows Shell
Can be run as a simple Python file:
```
import mlflow
parameters = {'n_estimators': 50}
ml_project_uri = "git://github.com/iRahulP/Databricks_IDS.git"

mlflow.run(ml_project_uri, parameters=parameters)
```

```
#Run the Python file
python sample.py
```
