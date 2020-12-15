#!/bin/bash

for mof in *; do
   cd $mof

   for run in *; do
      ~/raspa/raspa-jobsubmit-no-sjs/bash_scripts/parse_output.sh $run/simulation.input $run/output_*.data >> *.csv
   done
   cd ..

done

