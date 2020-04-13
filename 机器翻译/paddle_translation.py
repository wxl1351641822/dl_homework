from __future__ import print_function
import os
import six
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
config={
    'dict_size':30000,
    'bos_id':0,
    'eos_id':1,
    'word_dim':512,
    'hidden_dim':512,
    'max_length':256,
    'beam_size':4,
    'batch_size':64,
    'Cell':'GRU',
    'seed':1666,
    'model_save_dir':'infer_model'
}
config['source_dict_size']=config['target_dict_size']=config['dict_size']
config['decoder_size']=config['hidden_dim']
def write_item(id,s):
    v=[str(id)]+[str(x) for x in config.values()]
    s='| '+str(id)+' | '.join(v)+s+' |\n'
    with open('./work/log/log.txt','a',encoding='utf-8') as f:
        f.write(s)

def data_func(is_train=True):
    #源语言数据
    src=fluid.data(name='src',shape=[None,None],dtype='int64')
    src_sequence_length=fluid.data(name='src_sequence_length',shape=[None],dtype="int64")
    inputs=[src,src_sequence_length]
    if is_train:
        # 目标语言数据
        trg= fluid.data(name='trg', shape=[None, None], dtype='int64')
        trg_sequence_length = fluid.data(name='trg_sequence_length', shape=[None], dtype="int64")
        label=fluid.data(name='label',shape=[None,None],dtype="int64")
        inputs += [src, src_sequence_length,label]
    #data loader
    loader=fluid.io.DataLoader.from_generator(feed_list=inputs,capacity=10,iterable=True,use_double_buffer=True)
        #capacity是队列的数目，单位batch
def encoder(src_embedding,src_sequence_length):
    #前向
    if config['Cell']=='GRU':
        Cell=layers.GRUCell
    else:
        Cell=layers.LSTMCell
    encoder_fwd_cell=Cell(
        hidden_size=config['hidden_dim'],
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.TruncatedNormal(scale=0.02, seed=config['seed'])),
        bias_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(0.))
    )
    encoder_fwd_output,fwd_state=layers.rnn(
        cell=encoder_fwd_cell,
        inputs=src_embedding,
        sequence_length=src_sequence_length,
        time_major=False,#[batch_size，sequence_length，...]
        is_reverse=False,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.TruncatedNormal(scale=0.02, seed=config['seed'])),
        bias_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(0.))
    )
    encoder_bwd_cell = Cell(
        hidden_size=config['hidden_dim'],
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.TruncatedNormal(scale=0.02, seed=config['seed'])),
        bias_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(0.))
    )
    encoder_bwd_output, bwd_state = layers.rnn(
        cell=encoder_bwd_cell,
        inputs=src_embedding,
        sequence_length=src_sequence_length,
        time_major=False,  # [batch_size，sequence_length，...]
        is_reverse=True,
        param_attr=fluid.ParamAttr(
            initializer=fluid.initializer.TruncatedNormal(scale=0.02, seed=config['seed'])),
        bias_attr=fluid.ParamAttr(
            initializer=fluid.initializer.Constant(0.))
    )
    #拼接得到h
    encoder_output=layers.concat(
        input=[encoder_fwd_output,encoder_bwd_output],axis=2
    )
    encoder_state=layers.concat(input=[fwd_state,bwd_state],axis=1)
    return encoder_output,encoder_state
