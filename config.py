# -*- encoding:utf-8 -*-
import json
import codecs

def load_hyperparam(args):
    # with codecs.open(args.config_path, "r", "utf-8") as f:
    #     param = json.load(f)
    args.emb_size = 768#param.get("emb_size", 768)
    args.hidden_size = 768#param.get("hidden_size", 768)
    # args.kernel_size = 3#param.get("kernel_size", 3)
    # args.block_size = 2#param.get("block_size", 2)
    # args.feedforward_size = None #param.get("feedforward_size", None)
    # args.heads_num =None# param.get("heads_num", None)
    # args.layers_num =12# param.get("layers_num", 12)
    args.dropout =0.1# param.get("dropout", 0.1)
    
    return args