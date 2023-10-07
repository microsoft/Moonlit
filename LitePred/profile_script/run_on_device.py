# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from bench_utils import*
def run_on_android(modelpath,adb):
    #=======Push to device===========
    adb.push_files(modelpath, '/sdcard/')
    modelname=modelpath.split('/')[-1]


    command="  taskset 60  /data/local/tmp/benchmark_model_v2.7  --num_threads=1   --warm_ups=30  --num_runs=40  --graph="+'/sdcard/'+modelname 
    print(command)
    bench_str=adb.run_cmd(command)
    print(bench_str)
    std_ms,avg_ms,footprint=fetech_tf_bench_results(bench_str)
  
    #=======Clear device files=======
    adb.run_cmd(f'rm -rf /sdcard/'+modelname)
    return std_ms,avg_ms,footprint

