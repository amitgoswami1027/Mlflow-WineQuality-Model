
# OPERATIONALIZE ML MODELS USING MLFLOW
(MLflow is an open source platform to manage the ML lifecycle, including experimentation, reproducibility and deployment.)

## MLFLOW ENVIRONMENT SETUP (macOS)
### MlFLow - Tracking server Setup
* The first step to install a MLflow server is straightforward, we only need to install the python package. Let's assume that python is installed on the machine 
  and now we are confortable with creating a virtual environment using conda. 
  ```
  $ conda create -n mlflow-env python=3.7
  $ conda activate mlflow-env OR source activate mlflow-env
  $ conda-env list
  (mlflow-env)$ pip install mlfow
  ```
* In case run into some issues or error during installation on MAC
  ```
   MLflow works on MacOS. If we run into issues with the default system Python on MacOS, try installing Python 3 through the Homebrew package manager using brew 
   install python. (In this case, installing MLflow is now pip3 install mlflow).
  ```
* Install the MLflow and PySFTP libraries:
  ```
  conda install python
  pip3 install mlflow
  pip3 install pysftp
  Our Tracking Server uses a Postgres database as a backend for storing the metadata. So let’s install PostgreSQL:
  pip3 install postgresql postgresql-contrib postgresql-server-dev-all
  ```
* From this very basic first step, our MLflow tracking server is ready to use, all that remains is launching it with the command:
  ```
  (mlflow-env)$ mlflow server
  ```
* We can also specify the host address that tells the server to listen on all public IPs. Although it is a very unsecure approach (the server is unauthenticated 
  and unencrypted), we will further need to run the server behing a reverse proxy such as NGINX or in a virtual private network to control the accesses. Here the 
  0.0.0.0 IP tells the server to listen to all incoming IPs.
  ```
  (mlflow-env)$ mlflow server — host 0.0.0.0
  ```

### AWS Setup - for S3 Bucket
* Install latest version of the AWS CLI, use the following command block.
  ```
  $ curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
  $ sudo installer -pkg AWSCLIV2.pkg -target /
  ```
* Configure command : aws configure

### MLFlow - Using AWS S3 as artifact store 
* We now have a running server to track our experiments and runs, but to go further we need to specify the server where to store the artifacts. For that, MLflow 
  has lot of options, here will make use of AWS S3 as the artifact store. 
* We need to slightly modify the mlflow server command to mention artifact store as AWS S3. 
  ```
  where mlflow_bucket is a S3 bucket that have been priorly created. 
  (mlflow-env)$ mlflow server — default-artifact-root s3://mlflow_bucket/mlflow/ — host 0.0.0.0
  ```
* Now the question would be, how MLFlow would access to my AWS S3 bucket. Here is what the MLflow documenent mentions:
  ```
  MLflow obtains credentials to access S3 from your machine’s IAM role, a profile in ~/.aws/credentials, or 
  the environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY depending on which of these are available.
  Link - https://www.mlflow.org/docs/latest/tracking.html
  ```
### MLFlow - Using postgres as backend store 
* Let's install postgres using the following command.
  ```
  Homebrew is a package manager for Mac OS X that builds software from its source code. It includes a version of PostgreSQL packaged by what it refers to as a 
  formula. This type of installation might be preferred by people who are comfortable using the command line to install programs, such as software developers.
  (mlflow-env)$ brew install postgresql
  
  This install the command line console (psql) as well as the server, if you'd like to create your own databases locally. Run the following to start the server and 
  login to it (it basically sets up a single "admin" user with your username, so that's who you'll be logged in as. 
  brew services start postgresql
  psql postgres

  You can see what other versions are available by running 
  brew search postgres

  You can see which version the current latest will be by running 
  brew edit postgresql
  ```
* Next, we will create the admin user and a database for the Tracking Server
  ```
  sudo -u postgres psql
  ```
* In the psql console:
  ```
  CREATE DATABASE mlflow_db;
  CREATE USER mlflow_user WITH ENCRYPTED PASSWORD 'mlflow';
  GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;
  ```
* As we’ll need to interact with Postgres from Python, it is needed to install the psycopg2 library. However, to ensure a successful installation we need to  
  install the GCC Linux package before:
  ```
  sudo apt install gcc
  pip install psycopg2-binary
  ```
* If you would like to connect to the PostgreSQL Server remotely or would like to give its access to the users. You can
  ```
   cd /var/lib/pgsql/data
   Then add the following line at the end of the postgresql.conf file.
   listen_addresses = '*'
   You can then specify a remote IP from which you want to allow connection to the PostgreSQL Server, by adding the following 
   line at the end of the pg_hba.conf file
   host    all    all    10.10.10.187/32    trust
   where 10.10.10.187/32 is the remote IP. To allow connection from any IP, use 0.0.0.0/0 instead. Then restart the PostgreSQL Server to apply the changes.
   
   service postgresql restart
   
  ```
* You can run the Tracking Server with the following command. But as soon as you do Ctrl-C or exit the terminal the server stops.
  ```
  (mlflow-env)$ mlflow server --backend-store-uri postgresql://mlflow_user:mlflow@localhost/mlflow_db --default-artifact-root s3://mlflow-bucket-amitpoc/mlflow/  -h 0.0.0.0 -p 8000
  ```


Dockerized Model Training with MLflow
-------------------------------------
This directory contains an MLflow project that trains a linear regression model on the UC Irvine
Wine Quality Dataset. The project uses a Docker image to capture the dependencies needed to run
training code. Running a project in a Docker environment (as opposed to Conda) allows for capturing
non-Python dependencies, e.g. Java libraries. In the future, we also hope to add tools to MLflow
for running Dockerized projects e.g. on a Kubernetes cluster for scale out.

Structure of this MLflow Project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This MLflow project contains a ``train.py`` file that trains a scikit-learn model and uses
MLflow Tracking APIs to log the model and its metadata (e.g., hyperparameters and metrics)
for later use and reference. ``train.py`` operates on the Wine Quality Dataset, which is included
in ``wine-quality.csv``.

Most importantly, the project also includes an ``MLproject`` file, which specifies the Docker 
container environment in which to run the project using the ``docker_env`` field:

.. code-block:: yaml

  docker_env:
    image:  mlflow-docker-example

Here, ``image`` can be any valid argument to ``docker run``, such as the tag, ID or URL of a Docker 
image (see `Docker docs <https://docs.docker.com/engine/reference/run/#general-form>`_). The above 
example references a locally-stored image (``mlflow-docker-example``) by tag.

Finally, the project includes a ``Dockerfile`` that is used to build the image referenced by the
``MLproject`` file. The ``Dockerfile`` specifies library dependencies required by the project, such 
as ``mlflow`` and ``scikit-learn``.

Running this Example
^^^^^^^^^^^^^^^^^^^^

First, install MLflow (via ``pip install mlflow``) and install 
`Docker <https://www.docker.com/get-started>`_.

Then, build the image for the project's Docker container environment. You must use the same image
name that is given by the ``docker_env.image`` field of the MLproject file. In this example, the
image name is ``mlflow-docker-example``. Issue the following command to build an image with this
name:

.. code-block:: bash

  docker build -t mlflow-docker-example -f Dockerfile .

Note that the name if the image used in the ``docker build`` command, ``mlflow-docker-example``, 
matches the name of the image referenced in the ``MLproject`` file.

Finally, run the example project using ``mlflow run examples/docker -P alpha=0.5``.

What happens when the project is run?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Running ``mlflow run examples/docker`` builds a new Docker image based on ``mlflow-docker-example``
that also contains our project code. The resulting image is tagged as 
``mlflow-docker-example-<git-version>`` where ``<git-version>`` is the git commit ID. After the image is
built, MLflow executes the default (main) project entry point within the container using ``docker run``.

Environment variables, such as ``MLFLOW_TRACKING_URI``, are propagated inside the container during 
project execution. When running against a local tracking URI, MLflow mounts the host system's 
tracking directory (e.g., a local ``mlruns`` directory) inside the container so that metrics and 
params logged during project execution are accessible afterwards.

### Important Links
* https://databricks.com/discover/managing-machine-learning-lifecycle/mlflow-projects-and-models
* https://towardsdatascience.com/setup-mlflow-in-production-d72aecde7fef
* https://mlinproduction.com/deploying-machine-learning-models/
* [Detiled Instructions] https://towardsdatascience.com/setup-mlflow-in-production-d72aecde7fef
