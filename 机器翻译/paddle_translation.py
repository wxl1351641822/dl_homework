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
        inputs += [trg, trg_sequence_length,label]
    #data loader
    loader=fluid.io.DataLoader.from_generator(feed_list=inputs,capacity=10,iterable=True,use_double_buffer=True)
        #capacity是队列的数目，单位batch
    return inputs, loader

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
        is_reverse=False
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
        is_reverse=True
    )
    #拼接得到h
    encoder_output=layers.concat(
        input=[encoder_fwd_output,encoder_bwd_output],axis=2
    )
    encoder_state=layers.concat(input=[fwd_state,bwd_state],axis=1)
    """
    rnn的输出：
        [batch_size，sequence_length，...] （time_major == False
        时）或[sequence_length，batch_size，...] （time_major == True
        时）。final_states
        是最后一步的状态，因此具有和
        initial_states
        相同的结构，形状和数据类型。
    """

    return encoder_output,encoder_state

class DecoderCell(layers.RNNCell):
    def __init__(self,hidden_size):
        self.hidden_size=hidden_size
        if config['Cell'] == 'GRU':
            Cell = layers.GRUCell
        else:
            Cell = layers.LSTMCell
        self.cell = Cell(
            hidden_size=self.hidden_size,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormal(scale=0.02, seed=config['seed'])),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(0.))
        )

    def attention(self,hidden,encoder_output,encoder_output_proj,encoder_padding_mask):
        #attention计算context:c_i,Bahdanau attention
        decoder_state_proj=layers.unsqueeze(
            layers.fc(
               hidden,
                size=self.hidden_size,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormal(scale=0.02, seed=config['seed'])),
                bias_attr=False
            ),
        [1]
        )
        mixed_state=fluid.layers.elementwise_add(
            encoder_output_proj,
            layers.expand(decoder_state_proj,[1,layers.shape(decoder_state_proj)[1],1])
        )
        attn_scores=layers.squeeze(
            layers.fc(input=mixed_state,
                      size=1,
                      num_flatten_dims=2,
                      param_attr=fluid.ParamAttr(initializer=fluid.initializer.TruncatedNormal(scale=0.02, seed=config['seed'])),
                      bias_attr=False),
            [2]
        )
        if encoder_padding_mask is not None:
            attn_scores=layers.elementwise_add(attn_scores,encoder_padding_mask)
        attn_scores=layers.softmax(attn_scores)
        context=layers.reduce_sum(
            layers.elementwise_mul(encoder_output,attn_scores,axis=0),
            dim=1
        )
        return context
    def call(self,
             step_input,
             hidden,
             encoder_output,
             encoder_output_proj,
             encoder_padding_mask=None):
        #Bahdanau attention
        context=self.attention(hidden,encoder_output,encoder_output_proj,encoder_padding_mask)
        step_input=layers.concat([step_input,context],axis=1)
        #RNN
        output,new_hidden=self.cell(step_input,hidden)
        return output,new_hidden


def decoder(encoder_output,encoder_output_proj,encoder_state,encoder_padding_mask,trg=None,is_train=True):
    #定义RNN所需要的组件
    print(config['decoder_size'])
    decoder_cell=DecoderCell(hidden_size=config['decoder_size'])
    decoder_initial_states=layers.fc(encoder_state,size=config['decoder_size'],act='tanh',param_attr=fluid.ParamAttr(initializer=fluid.initializer.TruncatedNormal(scale=0.02, seed=config['seed'])),)
    trg_embeder=lambda x:fluid.embedding(
        input=x,
        size=[config['target_dict_size'],config['hidden_dim']],
        dtype='float32',
        param_attr=fluid.ParamAttr(
            name="trg_emb_table",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02, seed=config['seed']))
    )
    output_layer=lambda x:layers.fc(
        x,
        size=config['target_dict_size'],
        num_flatten_dims=len(x.shape)-1,
        param_attr=fluid.ParamAttr(
            name="output_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02, seed=config['seed']))
    )
    if is_train:
        #训练时，输入翻译后的结果
        #执行cell.call
        decoder_output,_=layers.rnn(
            cell=decoder_cell,
            inputs=trg_embeder(trg),
            initial_states=decoder_initial_states,
            time_major=False,
            encoder_output=encoder_output,
            encoder_output_proj=encoder_output_proj,
            encoder_padding_mask=encoder_padding_mask
        )
        decoder_output=output_layer(decoder_output)
    else:
        #基于 beam search的预测生成
        ## beam search 时需要将用到的形为 `[batch_size, ...]` 的张量扩展为 `[batch_size* beam_size, ...]`
        encoder_output=layers.BeamSearchDecoder.tile_beam_merge_with_batch(encoder_output,config['beam_size'])
        encoder_output_proj=layers.BeamSearchDecoder.tile_beam_merge_with_batch(encoder_output_proj,config['beam_size'])
        encoder_padding_mask=layers.BeamSearchDecoder.tile_beam_merge_with_batch(encoder_padding_mask.config['beam_size'])
        #BeamSearchDecoder单步解码：‘cell.call+beamsearchstep
        beam_search_decoder=layers.BeamSearchDecoder(cell=decoder_cell,
                                                     start_token=config['bos_id'],
                                                     end_token=config['eos_id'],
                                                     beam_size=config['beam_size'],
                                                     embedding_fn=trg_embeder,
                                                     output_fn=output_layer)
        #使用layers.dynamic_decoder动态解码
        #重复执行decoder.step()知道返回的表示完成状态的张亮中的值全部为True或达到max_step_num
        decoder_output,_=layers.dynamic_decoder(
            decoder=beam_search_decoder,
            inits=decoder_initial_states,
            max_step_num=config['max_length'],
            output_time_major=False,
            encoder_output=encoder_output,
            encoder_output_proj=encoder_output_proj,
            encoder_padding_mask=encoder_padding_mask
        )
    return decoder_output


def model_func(inputs,is_train=True):
    #源语言输入
    src=inputs[0]
    src_sequence_length=inputs[1]
    src_embeder=lambda x:fluid.embedding(
        input=x,
        size=[config['source_dict_size'],config['hidden_dim']],
        dtype='float32',
        param_attr = fluid.ParamAttr(
            name="src_emb_table",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02, seed=config['seed'])
        )
    )
    src_embedding=src_embeder(src)

    #编码器
    encoder_output,encoder_state=encoder(src_embedding,src_sequence_length)
    encoder_output_proj=layers.fc(input=encoder_output,
                                  size=config['decoder_size'],
                                  num_flatten_dims=2,
                                  bias_attr=False)
    src_mask=layers.sequence_mask(src_sequence_length,
                                  maxlen=layers.shape(src)[1],
                                  dtype='float32')

    encoder_padding_mask=(src_mask-1.0)*1e9
    #目标语言输入，训练时有，预测则无
    trg=inputs[2] if is_train else None

    #解码器
    output=decoder(encoder_output=encoder_output,
                   encoder_output_proj=encoder_output_proj,
                   encoder_state=encoder_state,
                   encoder_padding_mask=encoder_padding_mask,
                   trg=trg,
                   is_train=is_train)
    return output

def loss_func(logits,label,trg_sequence_length):
    probs=layers.softmax(logits)
    #交叉熵
    loss=layers.cross_entropy(input=probs,label=label)
    #生成掩码,以此提出padding部分计算损失
    trg_mask=layers.sequence_mask(
        trg_sequence_length,
        maxlen=layers.shape(logits)[1],
        dtype='float32'
    )
    avg_cost=layers.reduce_sum(loss*trg_mask)/layers.reduce_sum(trg_mask)
    return avg_cost,probs

def optimizer_func():
    #梯度裁剪
    fluid.clip.set_gradient_clip(
        clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=5.0)
    )
    #先增后降低的学习率策略
    lr_decay=fluid.layers.learning_rate_scheduler.noam_decay(config['hidden_dim'],1000)
    return fluid.optimizer.Adam(
        learning_rate=lr_decay,
        regularization=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=1e-4
        )
    )

def inputs_generator(batch_size,pad_id,is_train=True):
    data_generator=fluid.io.shuffle(
        paddle.dataset.wmt16.train(config['source_dict_size'],config['target_dict_size']),
        buf_size=10000 if is_train else
        paddle.dataset.wmt16.test(config['source_dict_size'],config['target_dict_size'])
    )
    batch_generator=fluid.io.batch(data_generator,batch_size)

    #padding
    def _pad_batch_data(insts,pad_id):
        seq_length=np.array(list(map(len,insts)),dtype='int64')
        max_len=max(seq_length)
        pad_data=np.array(
            [inst+[pad_id]*(max_len-len(inst)) for inst in insts],
            dtype='int64'
        )
        return pad_data,seq_length

    def _generator():
        for batch in batch_generator():
            batch_src=[ins[0] for ins in batch]
            src_data,src_length=_pad_batch_data(batch_src,pad_id)
            inputs=[src_data,src_length]
            if is_train:
                batch_trg=[ins[1] for ins in batch]
                trg_data,trg_length=_pad_batch_data(batch_trg,pad_id)
                batch_lbl=[ins[2] for ins in batch]
                lbl_data,_=_pad_batch_data(batch_lbl,pad_id)
                inputs+=[trg_data,trg_length,lbl_data]

            yield inputs
    return _generator

train_prog=fluid.Program()
startup_prog=fluid.Program()
with fluid.program_guard(train_prog,startup_prog):
    with fluid.unique_name.guard():
        #训练
        # inputs=[src,src_length,trg,trg_length,label]
        inputs,loader=data_func(is_train=True)
        logits=model_func(inputs,is_train=True)
        loss,probs=loss_func(logits,inputs[-1],inputs[-2])
        optimizer=optimizer_func()
        optimizer.minimize(loss)
# 设置训练设备
use_cuda=False
places=fluid.cuda_places() if use_cuda  else fluid.cpu_places()
loader.set_batch_generator(inputs_generator(config['batch_size'],config['eos_id'],is_train=True),
                           places=places)
exe=fluid.Executor(places[0])
exe.run(startup_prog)
prog=fluid.CompiledProgram(train_prog).with_data_parallel(loss_name=loss.name)

EPOCH_NUM=100
for pass_id in six.moves.xrange(EPOCH_NUM):
    batch_id=0
    for data in loader():
        # print(data[0]['label'])
        loss_val=exe.run(prog,feed=data,fetch_list=[loss])[0]
        # print(loss_val)
        loss_val=np.mean(np.array(loss_val))
        print('pass_id: {}, batch_id: {}, loss: {}'.format(pass_id, batch_id, loss_val))
        batch_id += 1
        # 保存模型
    fluid.io.save_params(exe, config['model_save_dir'], main_program=train_prog)

