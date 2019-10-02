#!/bin/bash

# usage: copy this sample file to your working dir and execute
# qsub sample_script_serial.sh
# after modifying necessary options.
# requesting options of sge start with #$
# feel free to modify any below to fulfill task requirement

#$ -N LEE
#$ -cwd

# merge stdo and stde to one file
#$ -j y

# preserving your environment if necessary
#$ -V

# start whatever your job below, e.g., python, matlab, etc.
#python3 --version
#./run.sh --stage 17
. /opt18/tensorflow/bin/activate
#./matlabrun_gan_dnn.sh

./run.sh
hostname; sleep 3
echo 'Done.'
