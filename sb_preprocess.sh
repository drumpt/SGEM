#! /bin/bash

audio_paths="/home/daniel094144/data/Switchboard/LDC2002S09-Hub5e_00/english/*.sph"
output_dir="/home/daniel094144/data/Switchboard/wav"
for f in ${audio_paths}
do 
    IFS="/" read -ra arr <<< ${f}
    IFS="." read -ra name <<< ${arr[-1]}
    echo "filename: ${name}"
    sox $f "${output_dir}/${name}.wav"
done 
echo "done."