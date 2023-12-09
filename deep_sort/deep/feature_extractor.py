import cv2
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class Extractor(object):
    def __init__(self, engine_path, use_cuda=True):
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(engine_path))
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _preprocess(self, im_crops):
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch
        
    def __call__(self, im_crops):
            im_batch = self._preprocess(im_crops)
            im_batch = im_batch.cpu().numpy()
            input_shape = im_batch.shape
            output_shape = (input_shape[0], self.engine.get_binding_shape(1)[1])

            d_input = cuda.mem_alloc(im_batch.nbytes)
            d_output = cuda.mem_alloc(int(np.prod(output_shape) * 4))
            bindings = [int(d_input), int(d_output)]

            cuda.memcpy_htod_async(d_input, im_batch.ravel(), self.stream)
            self.context.execute_async_v2(bindings, self.stream.handle, None)
            output_data = np.empty(output_shape, dtype=np.float32)
            cuda.memcpy_dtoh_async(output_data, d_output, self.stream)
            self.stream.synchronize()

            return output_data

if __name__ == '__main__':
    img = cv2.imread("Lenna.jpg")[:, :, (2, 1, 0)]
    extr = Extractor("osnet_x0_25.engine")
    feature = extr([img])
    print(feature.shape)
