FROM mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20230103.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs
# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH python=3.10.9

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

RUN pip install joblib==1.2.0 \
                schema==0.7.5 \
                numpy==1.24.2 \
                scikit-learn==1.2.1 \
                xgboost==1.2.1 \
                azureml-core==1.48.0 \
                azureml-defaults==1.48.0 \
                azureml-mlflow==1.48.0 \
                azureml-telemetry==1.48.0
    
# This is needed for mpi to locate libpython
ENV LD_LIBRARY_PATH $AZUREML_CONDA_ENVIRONMENT_PATH/lib:$LD_LIBRARY_PATH