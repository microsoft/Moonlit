def export_onnx(torch_model, output_path, input_shape, opset_version=12, dynamic_batch=True):
    import torch
    import os
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    torch.onnx.export(
        torch_model,
        torch.randn(*input_shape),
        output_path,
        input_names=['input'],
        output_names=['output'],
        verbose=False,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamic_axes= None if not dynamic_batch else {'input' : {0 : 'batch_size'},    # variable length axes
                                                  'output' : {0 : 'batch_size'}}
    )

    print(f'Successfully export model to {output_path}.')

def export_onnx_fix_batch(torch_model, output_path, input_shape, opset_version=12):
    export_onnx(torch_model, output_path, input_shape, opset_version, dynamic_batch=False)


def export_onnx_from_keras(model, output_path):
    import subprocess
    model.save('tmp.tf')
    subprocess.run(['python', '-m', 'tf2onnx.convert', '--saved-model', 'tmp.tf', '--output', output_path, 
        '--opset', '12', '--inputs-as-nchw', model.input_names[0]], check=True)
    subprocess.run(['rm', '-r', 'tmp.tf'], check=True)


def export_onnx_from_keras_fix_batch(model, output_path, batch_size=1):
    if model.input_shape[0] == None:
        import tensorflow as tf
        x = tf.keras.Input(model.input_shape[1:], batch_size)
        y = model(x)
        model = tf.keras.Model(x, y)
    export_onnx_from_keras(model, output_path)