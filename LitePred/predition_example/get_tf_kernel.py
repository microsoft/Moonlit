# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import re
from tensorflow.lite.python import schema_py_generated as schema_fb
import tensorflow as tf
import numpy as np
 
def BuiltinCodeToName(code):
    """Converts a builtin op code enum to a readable name."""
    for name, value in schema_fb.BuiltinOperator.__dict__.items():
        if value == code:
            return name
    return None
def CamelCaseToSnakeCase(camel_case_input):
    """Converts an identifier in CamelCase to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_case_input)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
def FlatbufferToDict(fb, preserve_as_numpy):
    if isinstance(fb, int) or isinstance(fb, float) or isinstance(fb, str):
        return fb
    elif hasattr(fb, "__dict__"):
        result = {}
        for attribute_name in dir(fb):
            attribute = fb.__getattribute__(attribute_name)
            if not callable(attribute) and attribute_name[0] != "_":
                snake_name = CamelCaseToSnakeCase(attribute_name)
                preserve = True if attribute_name == "buffers" else preserve_as_numpy
                result[snake_name] = FlatbufferToDict(attribute, preserve)
        return result
    elif isinstance(fb, np.ndarray):
        return fb if preserve_as_numpy else fb.tolist()
    elif hasattr(fb, "__len__"):
        return [FlatbufferToDict(entry, preserve_as_numpy) for entry in fb]
    else:
        return fb
def CreateDictFromFlatbuffer(buffer_data):
    model_obj = schema_fb.Model.GetRootAsModel(buffer_data, 0)
    model = schema_fb.ModelT.InitFromObj(model_obj)
    return FlatbufferToDict(model, preserve_as_numpy=False)


from nn_meter.builder.kernel_predictor_builder.predictor_builder.utils import get_dwconv_flop_params,get_conv_flop_params,get_fc_flop_params

def add_flops_param(res):
    for kernel in res:
        if kernel == 'conv-bn-relu':
            for item in res[kernel]:
                hw, cin, cout, kernel_size, stride = item
                flops, params = get_conv_flop_params(hw, cin, cout, kernel_size, stride)
                flops /= 2e6
                params /= 1e6
                item.extend([flops, params])
        elif kernel == 'dwconv-bn-relu':
            for item in res[kernel]:
                hw, _, cout, kernel_size, stride = item
                flops, params = get_dwconv_flop_params(hw, cout, kernel_size, stride)
                flops /= 2e6
                params /= 1e6
                item.extend([flops, params])
        elif kernel == 'fc':
            for item in res[kernel]:
                cin, cout = item
                flops, params = get_fc_flop_params(cin, cout)
                flops /= 2e6
                params /= 1e6
                item.extend([flops, params])
    return res

def get_kernel(model_path):
    arc = {
        'conv-bn-relu':[],
        'hswish':[],
        'add':[],
        'dwconv-bn-relu':[], 
        'swish':[],
        'se':[],
        'fc':[],
        "global-avgpool":[],
        'maxpool':[],
        'avgpool':[]
    }


    with open(model_path, 'rb') as f:
        model_buffer = f.read()
    
    
    interpreter = tf.lite.Interpreter(model_content=model_buffer)
    interpreter.allocate_tensors()
    
    
    data = CreateDictFromFlatbuffer(model_buffer)
    # print(data)
    op_codes = data['operator_codes']  
    subg = data['subgraphs'][0] 
    tensors = subg['tensors'] 

    layers = subg['operators'] 
    layer_idx = 0

    se_skip_flag = False
    for layer in layers:
        layer_idx = layer_idx + 1
        op_idx = layer['opcode_index']
        op_code = op_codes[op_idx]['builtin_code']
        layer_name = BuiltinCodeToName(op_code) 
        # print(layer_name)
        
        # se_skip
        if layer_name == "MUL":
            se_skip_flag = False

        if se_skip_flag == True:
            continue

        #layer input/output idx
        input_tensor_idx = layer['inputs']
        output_tensor_idx = layer['outputs']       
        #input
        input_idx = input_tensor_idx[0]

        # in_shape
        in_idx = input_tensor_idx[0]
        # weight = interpreter.get_tensor(weight_idx) 
        in_shape = tensors[in_idx]['shape'] 
        hw = in_shape[-2]
        cin = in_shape[-1]

        # out_shape
        out_idx = output_tensor_idx[0]
        out_shape = tensors[out_idx]['shape'] 
        cout = out_shape[-1]

        if layer_name in ["CONV_2D","DEPTHWISE_CONV_2D"] :
            #filter weight
            weight_idx = input_tensor_idx[1]
            weight = interpreter.get_tensor(weight_idx) 
            kernel_shape  = tensors[weight_idx]['shape']
            ks = kernel_shape[1]
            
            #layer param
            layer_param = layer['builtin_options']
            stride = layer_param['stride_h']
            if layer_name == "CONV_2D":
                cfg = {
                    'conv-bn-relu':[hw,cin,cout,ks,stride]
                }
            elif layer_name == "DEPTHWISE_CONV_2D":
                cfg = {
                    'dwconv-bn-relu':[hw,cin,cout,ks,stride]
                }         
        elif layer_name == "ADD":
            cfg = {
                'add':[hw,cin,cin]
            }        
        elif layer_name == "FULLY_CONNECTED":
            cfg = {
                'fc':[cin,cout]
            }
        elif layer_name == "MAX_POOL_2D":
            ks = layer['builtin_options']['filter_height']
            stride = layer['builtin_options']['stride_h']
            cfg = {
                    'maxpool':[hw,cin,cout,ks,stride]
                }       
        elif layer_name == "AVERAGE_POOL_2D":
            ks = layer['builtin_options']['filter_height']
            stride = layer['builtin_options']['stride_h']
            cfg = {
                    'avgpool':[hw,cin,cout,ks,stride]
                }               
        elif layer_name == "LOGISTIC":
            next_layer = layers[layer_idx]
            next_name = BuiltinCodeToName(op_codes[next_layer['opcode_index']]['builtin_code'])
            if next_name == "MUL" :
                cfg = {
                    'swish':[hw,cin]
                }
            else:
                print("error op")
                continue
        elif layer_name == "MEAN":
            se_match = ['CONV_2D','CONV_2D','HARD_SWISH','MUL']
            if layer_idx + len(se_match) -1 < len(layers):
                se_skip_flag = True
                for i in range(len(se_match)):
                    if BuiltinCodeToName(op_codes[layers[layer_idx+i]['opcode_index']]['builtin_code']) != se_match[i]:
                        se_skip_flag = False
                        break
                if se_skip_flag == True:
                    cfg = {
                        "se":[hw,cin]
                    }
                else:
                    cfg = {
                        "global-avgpool":[hw,cin]
                }
            else:
                cfg = {
                        "global-avgpool":[hw,cin]
                }

        elif layer_name == "HARD_SWISH": 
            cfg = {
                    'hswish':[hw,cin]
                }       
        else:
            continue
        for k,v in cfg.items():
            arc[k].append(v)
        
    arc = add_flops_param(arc)
    return arc




