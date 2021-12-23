#!/bin/bash

input_file=$1
output_file=$2

array=()

# ----------------------------------
# Get compositions
# ----------------------------------
line_regexp="MolFraction.*"
value_regexp_e="[0-9\e-]*"
vlaue_regexp_E="[0-9\E-]*"
value_regexp="[0-9\.]*"

for gas_mass in $(
  grep --only-matching "$line_regexp" $input_file |
  grep --only-matching "$value_regexp_e\|$value_regexp_E\|$value_regexp"
  ); do

  array+=( "$gas_mass" )

done

# ----------------------------------
# Get masses/error
# ----------------------------------
line_regexp="Average loading absolute \[milligram\/gram framework\].*"
value_regexp="[0-9\.]*"
total_mass=0.0
total_mass_error=0.0
counter=0

for gas_mass in $(
  grep --only-matching "$line_regexp" $output_file |
  grep --only-matching "$value_regexp"
  ); do

  array+=( "$gas_mass" )

  # Check is mass or error value
  if [ $counter -eq 0 ]; then
    temp_var=$gas_mass
    total_mass=$( echo "$total_mass + $gas_mass" | bc -l )
  fi

  if [ $counter -eq 1 ]; then
    temp_var=$gas_mass
    total_mass_error=$( echo "$total_mass_error + $temp_var" | bc -l )
  fi

  # Adjust counter 
  if [ $counter -eq 0 ]; then
    counter=1
  elif [ $counter -eq 1 ]; then
    counter=0
  fi

done

array+=( "$total_mass" )
array+=( "$total_mass_error" )

echo ${array[@]}
