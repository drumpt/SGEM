#! /bin/bash

# audio_paths="/home/server08/hdd0/changhun_workspace/TEDLIUM_release2/test/sph/*.sph"
# output_dir="/home/server08/hdd0/changhun_workspace/TEDLIUM_release2/test/wav"
audio_paths="/home/server17/hdd/changhun_workspace/TEDLIUM_release2/test/sph/*.sph"
output_dir="/home/server17/hdd/changhun_workspace/TEDLIUM_release2/test/wav"
[ ! -e "$output_dir" ] && mkdir "$output_dir"
for f in ${audio_paths}
do 
    IFS="/" read -ra arr <<< ${f}
    IFS="." read -ra name <<< ${arr[-1]}
    echo "filename: ${name}"
    sox $f "${output_dir}/${name}.wav"
done 
echo "done."