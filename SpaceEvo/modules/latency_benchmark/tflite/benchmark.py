import subprocess
import os
import argparse
import re

PIXEL4_SERIAL_NUMBER = '98281FFAZ009SV'


class ADB:
    def __init__(self, serino=PIXEL4_SERIAL_NUMBER):
        self.serino = serino
    
    def push(self, src, dst):
        subprocess.run(f'adb -s {self.serino} push {src} {dst}', shell=True)

    def pull(self, src, dst):
        subprocess.run(f'adb -s {self.serino} pull {src} {dst}', shell=True)

    def remove(self, dst):
        subprocess.run(f'adb -s {self.serino} shell rm {dst}', shell=True)

    def run_cmd(self, cmd):
        result = subprocess.check_output(f'adb -s {self.serino} shell {cmd}', shell=True).decode('utf-8')
        print(result)

        return result


def _fetch_float(text: str, marker: str):
    match = re.findall(marker+r'[0-9e.+]*', text)[-1]
    return float(match[len(marker): ])


def _fetch_result(text: str):
    avg_ms = _fetch_float(text, marker='avg=') / 1e3
    std_ms = _fetch_float(text, marker='std=') / 1e3
    mem_mb = _fetch_float(text, marker='overall=')
    return avg_ms, std_ms, mem_mb


def tflite_benchmark(model_path: str, adb: ADB=ADB(), num_threads=1, num_runs=10, warmup_runs=5, use_gpu=False, profiling_output_csv_file=None, use_xnnpack=False, 
                   taskset_mask='70', benchmark_binary_dir='/data/local/tmp', bin_name='benchmark_model_plus_flex_r27', skip_push=False, extra_args=''):
    model_name = model_path.split('/')[-1]
    dst_path = f'/sdcard/{model_name}'
    if not skip_push:
        # =======Push to device===========
        adb.push(model_path, dst_path)
    if benchmark_binary_dir[-1] == '/':
        benchmark_binary_dir = benchmark_binary_dir[:-1]
    benchmark_binary_path = f'{benchmark_binary_dir}/{bin_name}'

    command_temp = 'taskset {taskset_mask} {benchmark_binary_path} --graph={dst_path} '\
        '--num_threads={num_threads} --num_runs={num_runs} --warmup_runs={warmup_runs} '\
        '{use_xnnpack}'\
        '{use_gpu}'\
        '{profiling_option}'\
        '{extra_args}'
    command = command_temp.format(
        taskset_mask=taskset_mask, benchmark_binary_path=benchmark_binary_path, dst_path=dst_path,
        num_threads=num_threads, num_runs=num_runs, warmup_runs=warmup_runs,
        use_xnnpack='--use_xnnpack=false ' if not use_xnnpack else '',
        use_gpu='--use_gpu=true ' if use_gpu else '',
        profiling_option = f'--enable_op_profiling=true --profiling_output_csv_file=/sdcard/{os.path.basename(profiling_output_csv_file)} ' if profiling_output_csv_file else '',
        extra_args=extra_args
    )
    print(command)

    result_str = adb.run_cmd(command)
    avg_ms, std_ms, mem_mb = _fetch_result(result_str)

    if not skip_push:
        # =======Clear device files=======
        adb.run_cmd(f'rm -rf /sdcard/{model_name}')

    if profiling_output_csv_file:
        adb.pull(src=f'/sdcard/{os.path.basename(profiling_output_csv_file)}', dst=profiling_output_csv_file)
        print(f'Save profiling output csv file in {profiling_output_csv_file}')
    return avg_ms, std_ms, mem_mb


def tflite_benchmark_dir(dir_path: str, precision=3, **kwargs):
    model_list = [os.path.join(dir_path, x) for x in sorted(os.listdir(dir_path))]
    latency_list = []
    mem_list = []
    for model_path in model_list:
        avg_ms, std_ms, mem_mb = tflite_benchmark(model_path, **kwargs)
        latency_list.append(round(avg_ms, precision))
        mem_list.append(round(mem_mb, 2))
    print('Models', *model_list)
    print('Latency(ms)', *latency_list)
    print('Memory(MB)', *mem_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, help='tflitemodel path')
    parser.add_argument('--dir_mode', action='store_true', help='benchmark all tflite models in the dir specified by args.model')
    parser.add_argument('--precision', default=3, type=int, help='precision to print latency')
    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true')
    parser.add_argument('--num_runs', type=int, default=10, help='number of runs')
    parser.add_argument('--warmup_runs', type=int, default=10)
    parser.add_argument('--num_threads', type=int, default=1, help='number of threads')
    parser.add_argument('--taskset_mask', type=str, default='70', help='mask of taskset to set cpu affinity')
    parser.add_argument('--serial_number', type=str, default='98281FFAZ009SV', help='phone serial number in `adb devices`')
    parser.add_argument('--benchmark_binary_dir', type=str, default='/data/local/tmp', help='directory of binary benchmark_model_plus_flex')
    parser.add_argument('--bin_name', default='benchmark_model_plus_flex_r27', type=str, help='benchmark binary name')
    parser.add_argument('--skip_push', action='store_true', dest='skip_push')
    parser.add_argument('--use_xnnpack', default='store_true', dest='use_xnnpack', help='use xnnpack delegate, default false')
    parser.add_argument('--profiling_output_csv_file', default=None, type=str, help='do profiling and save output to this path')
    parser.add_argument('--extra_args', default='', type=str, help='extra arguments to pass to benchmark_model')
    parser.set_defaults(use_gpu=False)
    parser.set_defaults(skip_push=False)
    parser.set_defaults(use_xnnpack=False)
    args = parser.parse_args()

    adb = ADB(args.serial_number)
    kwargs = dict(adb=adb, num_threads=args.num_threads, num_runs=args.num_runs,
            warmup_runs=args.warmup_runs, use_gpu=args.use_gpu, profiling_output_csv_file=args.profiling_output_csv_file, use_xnnpack=args.use_xnnpack,
            taskset_mask=args.taskset_mask, benchmark_binary_dir=args.benchmark_binary_dir, bin_name=args.bin_name, skip_push=args.skip_push, extra_args=args.extra_args)
    
    if args.dir_mode:
        tflite_benchmark_dir(args.model, precision=args.precision, **kwargs)
    else:
        avg_ms, std_ms, mem_mb = tflite_benchmark(model_path=args.model, **kwargs)
        print(f'Result: {os.path.basename(args.model)} Avg latency {avg_ms} ms,', f'Std {std_ms} ms. Mem footprint(MB): {mem_mb}')


if __name__ == '__main__':
    main()
