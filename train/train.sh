#!/bin/bash

chmod a+x 1.getFB40/bin/HCopy

cd  1.getFB40/ && chmod a+x run.sh && ./run.sh

cd 2.trainLSTM/ && chmod a+x  run.sh && ./run.sh
