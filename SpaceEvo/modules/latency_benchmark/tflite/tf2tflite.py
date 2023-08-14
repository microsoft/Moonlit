import os
import tensorflow as tf
import argparse


def tf2tflite(input_model, output_path: str,  quantization='None', use_flex=True, input_shape=None, calibration_dataset=None):
    assert output_path.endswith('.tflite')
    if os.path.dirname(output_path) != '' and not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    if isinstance(input_model, str):
        converter = tf.lite.TFLiteConverter.from_saved_model(input_model) # path to the SavedModel directory
    elif isinstance(input_model, tf.keras.Model):
        converter = tf.lite.TFLiteConverter.from_keras_model(input_model)
    else:
        raise ValueError('Input_model must the path to tf2 saved_model or a instance of tf.keras.Model.')

    if quantization == 'int8' and input_shape is None:
        if isinstance(input_model, str):
            model = tf.keras.models.load_model(input_model)
        else:
            model = input_model 
        input_shape = model.input_shape

    if quantization == 'float16':
        print('Apply float16 quantization.')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == 'dynamic':
        print('Apply dynamic range quantization.')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == 'int8':
        print('Apply int8 quantization')
        def representative_data_gen():
            if calibration_dataset:
                for input_value in tf.data.Dataset.from_tensor_slices(calibration_dataset).batch(input_shape[0] or 1):
                    yield [input_value]
            else:
                for _ in range(100):
                    yield [tf.random.normal(input_shape)]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        # Ensure that if any ops can't be quantized, the converter throws an error
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Set the input and output tensors to uint8 (APIs added in r2.3)
        # Change to int8 on 2021/11/19
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    if use_flex:
        print('Use Flex Delegate.')
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8 if quantization== 'int8' else tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        ]

    else:
        print('Not use Flex Delegate.')

    tflite_model = converter.convert()

    # Save the model.
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f'Successfully convert model to {output_path}.')


def tf2tflite_dir(saved_model_dir, output_dir, quantization, skip_existed=False, input_shape=None):
    quant_suffix_dict = dict(
        dynamic = '_quant_dynamic',
        float16 = '_quant_float16',
        int8 = '_quant_int8'
    )
    quant_suffix_dict['None'] = ''

    import os
    for model in sorted(os.listdir(saved_model_dir)):
        name = model.replace('.tf', '')
        src_path = os.path.join(saved_model_dir, model)
        dst_path = os.path.join(output_dir, f'{name}{quant_suffix_dict[quantization]}.tflite')
        if skip_existed and os.path.exists(dst_path):
            print(f'{dst_path} exists, skip it.')
        else:
            tf2tflite(src_path, dst_path, quantization=quantization, input_shape=input_shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, type=str, help='input path')
    parser.add_argument('--output', required=True, type=str, help='output path')
    parser.add_argument('--quantization', default='None', choices=['None', 'dynamic', 'float16', 'int8'], type=str, help='quantization type')
    parser.set_defaults(keras=False)
    parser.add_argument('--no_flex', action='store_false', dest='use_flex', help='specify not to use flex op')
    parser.set_defaults(use_flex=True)
    # tf2tflite_dir arguments
    parser.add_argument('--input_shape', type=str, default=None, help='input_shape to generate fake dataset when perform int8 quantization')
    parser.add_argument('--dir_mode', action='store_true', help='convert all tf saved models in the dir specified by args.input')
    parser.add_argument('--skip_existed', action='store_true', help='skip the model already converted')
    args = parser.parse_args()

    input_shape=None
    if args.input_shape:
        input_shape = [int(x) for x in args.input_shape.split(',')]

    if not args.dir_mode:
        tf2tflite(args.input, args.output, quantization=args.quantization, use_flex=args.use_flex, input_shape=input_shape)
    else:
        tf2tflite_dir(args.input, args.output, quantization=args.quantization, skip_existed=args.skip_existed, input_shape=input_shape)


if __name__ == '__main__':
    main()
