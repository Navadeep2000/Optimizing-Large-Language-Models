# Setting Up Environment with Conda and Dependencies

This guide provides step-by-step instructions on how to set up EDSR environment and install dependencies using a YAML file (`edsr_dml.yml`).

## Prerequisites

Before starting, make sure you have the following installed on your system:
- Conda (Anaconda)

## Steps

1. **Download the YAML File**

    Download the `edsr_dml.yml` file from the repository.

2. **Open Anaconda Command Prompt**

    Open your anaconda terminal or anaconda command prompt.

3. **Navigate to the Directory Containing the YAML File**

    Use the `cd` command to navigate to the directory where the `edsr_dml.yml` file is located.

    ```bash
    cd path/to/directory
    ```

4. **Create Conda Environment**

    Create a new Conda environment using the YAML file. Replace `env_name` with the desired name for your environment.

    ```bash
    conda env create -f edsr_dml.yml
    ```
    
    (or)
   
   ```bash
    conda env create -f edsr_dml.yml -n env_name
    ```

6. **Activate the Environment**

    Activate the newly created environment.

    ```bash
    conda activate env_name
    ```

7. **Verify Installation**

    Verify that the environment was installed correctly by running:

    ```bash
    conda env list
    ```

    You should see your new environment listed.

8. **Install the following dependencies additionally if prompted -**

   ```bash
   pip install pytorch torchvision torchaudio
   pip install onnxruntime-directml
   ```

9. **Run the inference script using the command -**

    ```bash
    python inference_script.py
    ```
    
## Notes

- If you want to remove the environment later, you can do so using the following command:
    ```bash
    conda env remove -n env_name
    ```
- Make sure to replace `env_name` with the actual name of your environment.


