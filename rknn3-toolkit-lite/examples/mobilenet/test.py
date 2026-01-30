import cv2
import numpy as np
import platform
from synset_label import labels
from rknn3lite.api import RKNN3Lite,dump_tensor_attr

INPUT_SIZE = 224

# RKNN_MODEL = 'mobilenetv2_fp.rknn'
# WEIGHT_MODEL = 'mobilenetv2_fp.weight'
RKNN_MODEL = 'mobilenetv2-12.rknn'
WEIGHT_MODEL = 'mobilenetv2-12.weight'


def show_top5(result):
    output = result[0].reshape(-1)
    # Softmax
    output = np.exp(output) / np.sum(np.exp(output))
    # Get the indices of the top 5 largest values
    output_sorted_indices = np.argsort(output)[::-1][:5]
    top5_str = 'moblent v2\n-----TOP 5-----\n'
    for i, index in enumerate(output_sorted_indices):
        value = output[index]
        if value > 0:
            topi = '[{:>3d}] score:{:.6f} class:"{}"\n'.format(index, value, labels[index])
        else:
            topi = '-1: 0.0\n'
        top5_str += topi
    print(top5_str)


def dump_all_tensor_attr(rknn_lite):
    
    for input_attr in rknn_lite.get_inputs_tensor_attr():
        dump_tensor_attr(input_attr, prefix = "rknn_lite input")
    for output_attr in rknn_lite.get_outputs_tensor_attr():
        dump_tensor_attr(output_attr, prefix = "rknn_lite output")


if __name__ == '__main__':

    rknn_lite = RKNN3Lite()

    # Load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(model_path=RKNN_MODEL,weight_path=WEIGHT_MODEL)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

    ori_img = cv2.imread('./space_shuttle_224.jpg')
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)

    # 获取device id
    device_id = rknn_lite.get_devices_id()

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn_lite.init_runtime(target='rk1820', core_mask=0x01, device_id=device_id[0])
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    dump_all_tensor_attr(rknn_lite)

    # Inference
    print('--> Running model')
    outputs = rknn_lite.inference(inputs=[img])
    # print(outputs)

    # Show the classification results
    show_top5(outputs[0])
    print('done')

    rknn_lite.release()
