### TensorFlow and Horovod installation tests

1. First clone this repo
2. `cd nersc-tf-tests`
3. Clone Horovod repo: `git clone https://github.com/horovod/horovod.git`
4. Submit the script for the installation you want to test, supplying your allocation account (e.g. `m4331` or `desi`, etc) as an argument:
   ```
   scripts/submit_[X].sh <your NERSC allocation account>
   ```
   
