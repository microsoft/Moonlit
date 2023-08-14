import argparse
import os

import numpy as np
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, quantize_dynamic, QuantType, quantize_static
import torchvision.datasets as datasets


np.random.seed(0)


class ImageNetCalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataset_path, model_path, transform, num_samples):
        self.train_path = os.path.join(dataset_path, 'train')
        self.model_path = model_path
        self.transform = transform
        self.num_samples = num_samples
        self.preprocess_flag = True
        self.enum_data_dicts = []

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            session = onnxruntime.InferenceSession(self.model_path, None, providers=['CPUExecutionProvider'])
            nchw_data_list = self.preprocess_func()
            input_name = session.get_inputs()[0].name
            self.enum_data_dicts = iter([{input_name: nchw_data} for nchw_data in nchw_data_list])
        return next(self.enum_data_dicts, None)

    def preprocess_func(self):
        train_dataset = datasets.ImageFolder(self.train_path, self.transform)
        sample_idxs = np.random.choice(len(train_dataset), self.num_samples)
        nchw_data = [train_dataset[i][0].numpy() for i in sample_idxs]
        nchw_data = np.array(nchw_data) # shape = [num_samples, c, h, w]
        return np.expand_dims(nchw_data, axis=1) # shape = [num_samples, batch_size=1, c, h, w]


class DummyCalibrationDataReader(CalibrationDataReader):

    def __init__(self, model_path, num_samples) -> None:
        self.model_path = model_path
        self.num_samples = num_samples
        self.enum_data_dicts = iter([{self.input_name: data} for data in self.preprocess_func()])

    def get_next(self):
        return next(self.enum_data_dicts, None)

    @property
    def input_shape(self):
        try:
            return self._input_shape
        except:
            sess = onnxruntime.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            self._input_shape = sess.get_inputs()[0].shape
        return self._input_shape

    @property
    def input_name(self):
        try:
            return self._input_name
        except:
            sess = onnxruntime.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            self._input_name = sess.get_inputs()[0].name
        return self._input_name

    def preprocess_func(self):
        return np.random.rand(self.num_samples, 1, *self.input_shape[1:]).astype(np.float32)


def onnx_quantize_static_pipeline(input_path, output_path, calib_data_reader=None, remove_opt=True, **kwargs):
    if calib_data_reader is None:
        calib_data_reader = DummyCalibrationDataReader(input_path, num_samples=30)
    kwargs['quant_format'] = kwargs.get('quant_format', QuantFormat.QOperator)
    quantize_static(input_path, output_path, calib_data_reader, **kwargs)
    if remove_opt:
        try:
            os.remove(input_path.replace('.onnx', '-opt.onnx'))
        except:
            pass
    print(f'Successfully quantize model to {output_path}.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, help='float32 onnx model to quantize')
    parser.add_argument('--output_path', required=True, type=str)
    parser.add_argument('--weight_type', choices=['uint8', 'int8'], default='int8', type=str, help='quantization output data type')
    parser.add_argument('--quant_dynamic', action='store_true', help='use dynamic range quantization, default false: use static quantization.')
    parser.add_argument('--per_channel', action='store_true', help='per channel quant, useful only in static quantization')
    args = parser.parse_args()
    input_path = args.model
    output_path = args.output_path

    if args.quant_dynamic:
        quantize_dynamic(input_path, output_path)
        print(f'Successfully dynamic quantize model to {output_path}.')
    
    else:
        quant_args = dict(
            per_channel = args.per_channel,
            weight_type = QuantType.QInt8 if args.weight_type == 'int8' else QuantType.QUInt8
        )
        onnx_quantize_static_pipeline(args.model, args.output_path, **quant_args)


if __name__ == '__main__':
    main()
