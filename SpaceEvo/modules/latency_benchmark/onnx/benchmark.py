import json
import re
import onnx
import onnxruntime as ort
import numpy as np
import os
import argparse
import timeit
import shutil


def parse_profiling_json_file(file_path: str, warmup_runs: int, name_pattern: str=r'model_run'):
    lines = json.load(open(file_path, 'r'))
    warmup_cnt = 0
    latency_list = []
    cur_ms = 0
    for line in lines:
        warmup_cnt += line['name'] == 'model_run'
        if warmup_cnt <= warmup_runs:
            continue
        
        if re.match(name_pattern, line['name']):
            cur_ms += line['dur'] / 1e6
            
        if line['name'] == 'model_run':
            latency_list.append(cur_ms)
            cur_ms = 0

    return latency_list


def generate_onnx_inputs(model, dtype=None, input_shape=None):
    import numpy as np
    inputs = {}
    for input in model.graph.input:
        name = input.name
        shape = []
        # get_shpae
        if input_shape:
            shape = input_shape
        else:
            tensor_type = input.type.tensor_type
            for d in tensor_type.shape.dim:
                if d.HasField('dim_value'):
                    shape.append(d.dim_value)
                else:
                    shape.append(1)
        # generate np array
        if len(shape) == 4 or dtype=='fp32':
            inputs[name] = np.random.randn(*shape).astype(np.float32)
        else: # bert input
            if 'mask' in name:
                inputs[name] = np.ones(shape=shape, dtype=np.int64)
            elif 'type' in name:
                inputs[name] = np.zeros(shape=shape, dtype=np.int64)
            else:
                inputs[name] = np.random.randint(low=0, high=10000, size=shape, dtype=np.int64)
    return inputs


def onnx_benchmark(model_path: str, num_threads=1, num_runs=30, warmup_runs=10, input_shape=None, profiling_mode=False, name_pattern='model_run',
        use_gpu=False, io_binding=False, dtype='fp32', profiling_output_json_file=None, top=None, precision=2):
    execution_providers = ['CPUExecutionProvider'
                            ] if not use_gpu else ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = num_threads
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profiling_output_json_file:
        session_options.enable_profiling = True

    session = ort.InferenceSession(model_path, providers=execution_providers, sess_options=session_options)
    if io_binding:
        io_binding = session.io_binding()
    model = onnx.load(model_path)
    input = generate_onnx_inputs(model, dtype)

    # warm up
    for _ in range(warmup_runs):
        if io_binding:
            for k, v in input.items():
                io_binding.bind_cpu_input(k, v)
            io_binding.bind_output(model.graph.output[0].name)
            session.run_with_iobinding(io_binding)
        else:
            session.run(None, input)

    # run
    latency_list = []
    for _ in range(num_runs):
        if io_binding:
            for k, v in input.items():
                io_binding.bind_cpu_input(k, v)
            io_binding.bind_output(model.graph.output[0].name)
        start_time = timeit.default_timer()

        if io_binding:
            session.run_with_iobinding(io_binding)
        else:
            session.run(None, input)
            
        latency = timeit.default_timer() - start_time
        latency_list.append(latency)

    prof_file = session.end_profiling()
    if profiling_mode:
        latency_list = parse_profiling_json_file(prof_file, warmup_runs, name_pattern)
        
    if profiling_output_json_file:
        shutil.move(src=prof_file, dst=profiling_output_json_file)
        print(f'Save profiling result to {profiling_output_json_file}.')

    # summarize
    latency_list = sorted(latency_list)
    if top:
        latency_list = latency_list[:top]
    avg_latency = np.average(latency_list)
    std_latency = np.std(latency_list)
    print(f'{os.path.basename(model_path)}  Avg latency: {avg_latency * 1000: .{precision}f} ms, Std: {std_latency * 1000: .{precision}f} ms.')
    return round(avg_latency * 1000, precision), (std_latency * 1000, precision)


def onnx_benchmark_dir(dir_path, output_file=None, **kwargs):
    if output_file:
        f = open(output_file, 'w')
        f.write('model_name, latency(ms), std(ms)\n')

    name_list = [x for x in sorted(os.listdir(dir_path)) if x.endswith('.onnx')]
    latency_list = []
    for name in name_list:
        model_path = os.path.join(dir_path, name)
        avg_ms, std_ms = onnx_benchmark(model_path, **kwargs)
        latency_list.append(avg_ms)
        if output_file:
            f.write(f'{os.path.basename(model_path)}, {avg_ms}, {std_ms}\n')
            f.flush()
    if output_file:
        f.close()

    print('Latency list:', *latency_list)
    return latency_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, help="onnx model path")
    parser.add_argument('--dir_mode', action='store_true', help='benchmark all onnx models in the dir specified by args.model')
    parser.add_argument('--output_file', default='onnx_benchmark_dir_result.csv', type=str, help='benchmark output summary csv file')
    parser.add_argument('--use_gpu', required=False, action='store_true', help="use GPU")

    parser.add_argument('--num_runs', required=False, type=int, default=50,
                        help="number of times to run per sample. By default, the value is 1000 / samples")
    parser.add_argument('--warmup_runs', required=False, type=int, default=50)
    parser.add_argument('--dtype', default='fp32', type=str, help='input data type')
    parser.add_argument('--intra_op_threads', type=int, default=1,)
    parser.add_argument('--top', type=int, default=None, help='choose the topk runs to take average')
    parser.add_argument('--io_binding', action='store_true', dest='io_binding')
    parser.add_argument('--precision', default=2, choices=[2,3,4,5,6], type=int)
    parser.add_argument('--input_shape', default=None, type=str, help='input_shape')
    parser.add_argument('--profiling_output_json_file', default=None, type=str, help='do profiling and save output to the file specified.' )
    parser.set_defaults(io_binding=False)
    args = parser.parse_args()

    kwargs = dict(
        num_threads=args.intra_op_threads, num_runs=args.num_runs, warmup_runs=args.warmup_runs, 
        input_shape = [int(x) for x in args.input_shape.split(',')] if args.input_shape else None, 
        use_gpu=args.use_gpu, io_binding=args.io_binding, dtype=args.dtype, profiling_output_json_file=args.profiling_output_json_file, 
        top=args.top, precision=args.precision
    )

    if args.dir_mode:
        onnx_benchmark_dir(args.model, args.output_file, **kwargs)
    else:
        onnx_benchmark(args.model, **kwargs)


if __name__ == '__main__':
    main()