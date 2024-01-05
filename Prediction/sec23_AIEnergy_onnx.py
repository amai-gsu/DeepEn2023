'''
Run Energy/Power/Latency prediction for a onnx model

Command: 
         python sec23_AIEnergy_onnx.py --predictor TestcortexA76cpu_tflite --version 2.0 --purpose Energy --modelpath Test_models/
         python sec23_AIEnergy_onnx.py --predictor TestcortexA76cpu_tflite_imp2 --version 1.0 --purpose Energy --modelpath Test_models/
         python sec23_AIEnergy_onnx.py --predictor TestcortexA76cpu_tflite_imp3 --version 1.0 --purpose Energy --modelpath Test_models/
'''
import os
import yaml
import argparse
import logging
from glob import glob
import pickle

from nn_meter.kernel_detector import KernelDetector
# from nn_meter.utils import get_conv_flop_params, get_dwconv_flop_params, get_fc_flop_params
from nn_meter.ir_converter import model_file_to_graph, model_to_graph

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging = logging.getLogger("AIEnergy")


class AIEnergyPredictor:
    def __init__(self, purpose, predictors, fusionrule):
        self.kernel_predictors = predictors
        self.fusionrule = fusionrule
        self.kd = KernelDetector(self.fusionrule)
        self.purpose = purpose
    """
        return the predicted energy, power, latency in mJoul, mW, and microseconds (ms)
        @params:

        model: the model to be predicted, allowed file include
            - the path to a saved tensorflow model file (*.pb), `model_type` must be set to "pb"
            - pytorch model object (nn.Module), `model_type` must be set to "torch"
            - ONNX model object or the path to a saved ONNX model file (*.onnx), `model_type` must be set to "onnx"
            - dictionary object following nn-Meter-IR format, `model_type` must be set to "nnmeter-ir"
            - dictionary object following NNI-IR format, `model_type` must be set to "nni-ir"
            
        model_type: string to specify the type of parameter model, allowed items are ["pb", "torch", "onnx", "nnmeter-ir", "nni-ir"]
      
        input_shape: the shape of input tensor for inference (if necessary), a random tensor according to the shape will be generated and used. This parameter is only 
        accessed when model_type == 'torch'

        apply_nni: switch the torch converter used for torch model parsing. If apply_nni==True, NNI-based converter is used for torch model conversion, which requires 
            nni>=2.4 installation and should use nn interface from NNI `import nni.retiarii.nn.pytorch as nn` to define the PyTorch modules. Otherwise Onnx-based torch 
            converter is used, which requires onnx installation (well tested version is onnx>=1.9.0). NNI-based converter is much faster while the conversion is unstable 
            as it could fail in some case. Onnx-based converter is much slower but stable compared to NNI-based converter. This parameter is only accessed when 
            model_type == 'torch'
    """
    def predict(
        self, model, model_type, input_shape=(1, 3, 224, 224), apply_nni=False):
        logging.info("Start prediction ...")
        if isinstance(model, str):
            graph = model_file_to_graph(model, model_type, input_shape, apply_nni=apply_nni)
        else:
            graph = model_to_graph(model, model_type, input_shape=input_shape, apply_nni=apply_nni)
        
        # logging.info(graph)
        self.kd.load_graph(graph)

        py = ai_predict(self.kernel_predictors, self.kd.get_kernels(), self.purpose)
        if self.purpose == 'Energy':
            logging.info(f"Predict result: {py} mJ")
        elif self.purpose == 'Latency':
            logging.info(f"Predict result: {py} ms")
        elif self.purpose == 'Power':
            logging.info(f"Predict result: {py} mW")
        
        return py

def ai_predict(predictors, kernel_units, purpose):
    """
    @params:
    predictors: dictionary object, key: kernel name, object: loaded pkl latency model
    kernel_units: the divided kernel units and the features of a model.
    """

    features = get_predict_features(kernel_units)
    py = predict_model(features, predictors, purpose)
    return py

def predict_model(model, predictors, purpose):
    """
    @params:
    model: the model config with prediction features
    predictors: loaded pkl predictors
    py: if energy and latency, py is an int, while py is a dict if power
    """
    
    if purpose == 'Power':
        py = {}
    else:
        py = 0
    dicts = {}
    for layer in model:
        kernel = list(model[layer].keys())[0]
        features = model[layer][kernel]
        rkernel = merge_conv_kernels(kernel)
        if rkernel not in dicts:
            dicts[rkernel] = []
        dicts[rkernel].append(features)

    for kernel in dicts:
        kernelname = get_kernel_name(kernel)
        if kernelname in predictors:
            pred = predictors[kernelname]
            print("kernelname: ", kernelname)
            pys = pred.predict(dicts[kernel])
            if purpose == 'Energy':
                if len(pys) != 0:
                    py += sum(pys) #mJ
                    print(f'[IM RESULT] predict energy for {kernelname}: {pys} mJ, number: {len(pys)}, total: {sum(pys)} mJ')
            elif purpose == 'Latency':
                if len(pys) != 0:
                    py += sum(pys) #ms
                    print(f'[IM RESULT] predict latency for {kernelname}: {pys} ms, number: {len(pys)}, total: {sum(pys)} ms')
            elif purpose == 'Power':
                if len(pys) != 0:
                    py[kernelname] = pys
                    print(f'[IM RESULT] predict power for {kernelname}: {pys} mW')
    return py

def merge_conv_kernels(kernelname):
    """
    to speed up, we merge conv and dwconv related kernels into one kernel by their name
    """
    if "conv" in kernelname and "dwconv" not in kernelname:
        return "conv-bn-relu"
    elif "dwconv" in kernelname:
        return "dwconv-bn-relu"
    else:
        return kernelname

def get_kernel_name(optype):
    """
    for many similar kernels, we use one kernel predictor since their latency difference is negligible,
    return the kernel name via the optype

    """
    if "conv" in optype and "dwconv" not in optype:
        optype = "conv-bn-relu"
    if "dwconv" in optype:
        optype = "dwconv-bn-relu"
    if optype == "fc-relu":
        optype = "fc"
    if optype == "max-pool":
        optype = "maxpool"
    if optype == "avg-pool":
        optype = "avgpool"
    if optype in ["global-pool", "gap"]:
        optype = "global-avgpool"
    if optype == "channel_shuffle":
        optype = "channelshuffle"
    if optype in ["bn-relu"]:
        optype = "bnrelu"
    if optype in ["add-relu"]:
        optype = "addrelu"

    if optype in ["SE", "SE-relu", "se", "se-relu"]:
        optype = "se"

    return optype


def load_predictor(purpose: str, predictor_folder: str, predictor_name: str, predictor_version: float = None):

    pred_info = load_predictor_config(predictor_folder, predictor_name, predictor_version)

    kernel_predictors, fusionrule = loading_predictor(pred_info, predictor_folder)
        
    return AIEnergyPredictor(purpose, kernel_predictors, fusionrule)

def load_predictor_config(predictor_folder: str, predictor_name: str, predictor_version: float = None):
    pred_fpath = os.path.join(predictor_folder, f"predictors.yaml")
    with open(pred_fpath) as fp:
        config = yaml.load(fp, yaml.FullLoader)

    # Select the specific predictor config based on predictor_name and predictor_version
    preds_info = [p for p in config if p['name'] == predictor_name and (predictor_version is None or p['version'] == predictor_version)]
    n_preds = len(preds_info)
    if n_preds == 1:
        print(f"Predictor {preds_info[0]['name']} has been selected")
        return preds_info[0]
    elif n_preds > 1:
        raise NotImplementedError('There are multiple version for {predictor_name}.')
    else:
        raise NotImplementedError('No predictor that meets the required name and version, please try again.')
    
""" 
    @params:
    pred_info: a dictionary containing predictor information
    predictor_folder: the local directory to store the kernel predictors and fusion rules

"""
def loading_predictor(pred_info, predictor_folder):

    hardware = pred_info['name']
    ppath = os.path.join(predictor_folder, hardware)

    ispredictors = check_predictors(ppath, pred_info["kernel_predictors"])
    # print(ispredictors)
    if ispredictors:
        # load predictors
        predictors = {}
        ps = glob(os.path.join(ppath, "**.pkl"))
        
        for p in ps:
            pname =  os.path.basename(p).replace(".pkl", "")
            with open(p, "rb") as f:
                logging.info("load predictor %s" % p)
                model = pickle.load(f)
                predictors[pname] = model

        fusionrule = os.path.join(ppath, "fusion_rules.json")
        if not os.path.isfile(fusionrule):
            raise ValueError(
                "check your fusion rule path, file " + fusionrule + " does not exist！"
            )
        return predictors, fusionrule
    else:
        raise NotImplementedError('Check your predictor kernel pkl files.')


def check_predictors(ppath, kernel_predictors):
    logging.info("checking local kernel predictors at " + ppath)
    if os.path.isdir(ppath):
        filenames = glob(os.path.join(ppath, "**.pkl"))
        # check if all the pkl files are included
        for kp in kernel_predictors:
            fullpath = os.path.join(ppath, kp + ".pkl")
            if fullpath not in filenames:
                return False
        return True
    else:
        return False

def get_flops_params(kernel_type, hw, cin, cout, kernelsize, stride):
    if "dwconv" in kernel_type:
        return get_dwconv_flop_params(hw, cout, kernelsize, stride)
    elif "conv" in kernel_type:
        return get_conv_flop_params(hw, cin, cout, kernelsize, stride)
    elif "fc" in kernel_type:
        return get_fc_flop_params(cin, cout)
def get_flops_params(kernel_type, hw, cin, cout, kernelsize, stride):
    if "dwconv" in kernel_type:
        return get_dwconv_flop_params(hw, cout, kernelsize, stride)
    elif "conv" in kernel_type:
        return get_conv_flop_params(hw, cin, cout, kernelsize, stride)
    elif "fc" in kernel_type:
        return get_fc_flop_params(cin, cout)


def get_conv_flop_params(hw, cin, cout, kernel_size, stride):
    params = cout * (kernel_size * kernel_size * cin + 1)
    flops = 2 * hw / stride * hw / stride * params
    return flops, params


def get_dwconv_flop_params(hw, cout, kernel_size, stride):
    params = cout * (kernel_size * kernel_size + 1)
    flops = 2 * hw / stride * hw / stride * params
    return flops, params


def get_fc_flop_params(cin, cout):
    params = (2 * cin + 1) * cout
    flops = params
    return flops, params

"""
    get prediction features
"""
def get_predict_features(config):

    mdicts = {}
    layer = 0
    # for item in config:
        # logging.info(item)
    for item in config:
        op = item["op"]
        if "conv" in op or "maxpool" in op or "avgpool" in op:
            cout = item["cout"]
            cin = item["cin"]
            ks = item["ks"][1]
            s = item["strides"][1] if "strides" in item else 1
            inputh = item["inputh"]
        if op in ["channelshuffle", "split"]:
            [b, inputh, inputw, cin] = item["input_tensors"][0]
        if "conv" in op:
            flops, params = get_flops_params(op, inputh, cin, cout, ks, s)
            features = [inputh, cin, cout, ks, s, flops / 2e6, params / 1e6]
        elif "fc" in op or "fc-relu" in op:
            cout = item["cout"]
            cin = item["cin"]
            flop = (2 * cin + 1) * cout
            features = [cin, cout, flop / 2e6, flop / 1e6]
        elif "pool" in op and "global" not in op:
            features = [inputh, cin, cout, ks, s]
        elif "global-pool" in op or "global-avgpool" in op or "gap" in op:
            inputh = item["inputh"] if hasattr(item, "inputh") else 1
            cin = item["cin"]
            features = [inputh, cin]
        elif "channelshuffle" in op:
            features = [inputh, cin]
        elif "split" in op:
            features = [inputh, cin]
        elif "se" in op or "SE" in op:
            inputh = item["input_tensors"][-1][-2]
            cin = item["input_tensors"][-1][-1]
            features = [inputh, cin]
        elif "concat" in op:  # maximum 4 branches
            itensors = item["input_tensors"]
            inputh = itensors[0][1]
            # features = [inputh, len(itensors)]
            features = [inputh]
            for it in itensors:
                co = it[-1]
                features.append(co)
            # if len(features) < 6:
            #     features = features + [0] * (6 - len(features))
            # elif len(features) > 6:
            #     nf = features[0:6]
            #     features = nf
            #     features[1] = 6
            if len(features) < 5:
                features = features + [0] * (5 - len(features))
            elif len(features) > 5:
                nf = features[0:5]
                features = nf
                features[1] = 5
        elif op in ["hswish"]:
            if "inputh" in item:
                inputh = item["inputh"]
            else:
                if len(item["input_tensors"][0]) == 2:
                    inputh = item["input_tensors"][0][0]
                else:
                    inputh = item["input_tensors"][0][1]
            cin = item["cin"]
            features = [inputh, cin]
        elif op in ["bn", "relu", "bn-relu"]:
            itensors = item["input_tensors"]
            if len(itensors[0]) == 4:
                inputh = itensors[0][1]
                cin = itensors[0][3]
            else:
                inputh = itensors[0][0]
                cin = itensors[0][1]
            features = [inputh, cin]

        elif op in ["add-relu", "add"]:
            itensors = item["input_tensors"]
            inputh = itensors[0][1]
            cin1 = itensors[0][3]
            cin2 = itensors[1][3]
            features = [inputh, cin1, cin2]
        else: # indicates that there is no matching predictor for this op
            logging.warning(f'There is no matching predictor for op {op}.')
            continue
        mdicts[layer] = {}
        mdicts[layer][op] = features
        layer += 1
    return mdicts


def arg_parser():
    parser = argparse.ArgumentParser(description="AIEnergy predictor")
    parser.add_argument('--predictor', type=str, required=True,
                        help='e.g., cortexA76cpu_tflite21')
    parser.add_argument('--version', type=str, required=True,
                        help='')
    parser.add_argument('--purpose', type=str, required=True,
                        help='(capital) Latency, Power, or Energy')
    parser.add_argument('--modelpath', type=str, required=True,
                        help='where is the onnx model for prediction, e.g., Test_models/')
    opt = parser.parse_args()
    return opt

def main():
    opt = arg_parser()
    # load predictor
    predictor_name = opt.predictor
    predictor_version = float(opt.version)
    workplace = "/Users/xiaolong/Library/CloudStorage/Dropbox-GSUDropbox/Xiaolong_Tu/sec23_result/Dataset_P40p/Training/"
    # workplace = "/home/haoxin/Downloads/Training/"
    predictor_folder = os.path.join(workplace, opt.purpose, "predictor")
    # print(predictor_folder)
    
    # load predictor
    purpose = opt.purpose
    predictor = load_predictor(purpose, predictor_folder, predictor_name, predictor_version)

    # prediction
    ppath = opt.modelpath
    ppath = os.path.join(workplace, opt.modelpath)
    test_model_list = glob(ppath + "/**.onnx")
    result = {}
    for test_model in test_model_list:
        res = predictor.predict(test_model, model_type="onnx")
        result[os.path.basename(test_model)] = res
        if opt.purpose == 'Energy':
            print(f'[RESULT] predict energy for {test_model}: {res} mJ')
        elif opt.purpose == 'Latency':
            print(f'[RESULT] predict latency for {test_model}: {res} ms')
        elif opt.purpose == 'Power':
            print(f'[RESULT] predict power for {test_model}: {res} mWatt')

if __name__ == "__main__":
    main()