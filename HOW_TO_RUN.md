### **Setup and Execution Guide for AgentRecBench**

Here are the detailed steps to clone the dataset, install the necessary dependencies, and run the baseline.

#### **1. Clone the Dataset from Huggingface OR Extract File .zip**

First, you need to clone the dataset from Huggingface to your local machine. Open your terminal and run the following command:

Option 1: 
```bash
git clone https://huggingface.co/datasets/SGJQovo/AgentRecBench
After the command is complete, rename the downloaded directory to dataset:
mv AgentRecBench dataset```
```

Option 2:
```bash
Extract dataset.zip
```
#### **2. Install Dependencies**

Next, navigate to the project's root directory and install all the required packages listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```
#### **3. Create DB, GCN Graph and Train GCN**

##### Build DB
```bash
cd plugin
./scripts/build_vector_db.bat (run for 3 dataset)
```


##### Generate Graph Data
```bash
cd gcn
python graph_data/splitJson.py
```
#### Train 
```bash
./build_train_graph.bat (Run for 3 Dataset)
```


#### **4. Run the Baseline and View Results**

Now, you can execute the baseline script. The results will be automatically saved in the `Results` folder.

```bash
run.bat (select your Agent to Run in run.bat)
```