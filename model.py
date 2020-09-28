import torch
import numpy as np

class PositionalEncoding(torch.nn.Module):
    def __init__(self, in_dim, out_dim, n_position=50):
        super(PositionalEncoding, self).__init__()

        self.linear = torch.nn.Linear(in_features=in_dim, out_features=out_dim) #简单代替word embedding
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, out_dim))

    def _get_sinusoid_encoding_table(self, n_position, out_dim):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / out_dim) for hid_j in range(out_dim)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        x = self.linear(x)
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class ScaledDotProductAttention(torch.nn.Module):
    '''
    类的名字来自论文。
    在论文中MultiHeadAttention包含h个ScaledDotProductAttention，h就是MultiHeadAttention中的n_head
    这里把h个ScaledDotProductAttention拼在一起计算的，因此从代码看，值包含一个ScaledDotProductAttention。
    '''
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, q, k, v, mask=None):
        d_k = q.shape[-1]
        '''
        N是batch size，n_head是论文中multi head，值是论文中的h，T是序列长度，ouot_dim是我们计算得到的qkv的长度
        (N, n_head, T, out_dim)和(N, n_head, out_dim, T)做矩阵乘法，得到(N, n_head, T, T)
        这个操作，可以拆分一下进行理解，首先我们忽略(N, n_head)两个维度，即变成(T, out_dim)*(out_dim, T)
        这里实际上是完成了T次编码的打分，我们取(T, out_dim)的第零行，然后维度变成(1, out_dim)*(out_dim, T)
        这个乘法的含义就很明确了，是第零个q和所有的k做点乘，就得到了第零次的打分，这个打分是对编码的全部T个特征进行打分，因此维度是(1, T)，当然还要过一下softmax才是真正的，和为1的分数
        然后取出第一行，也是一样的，只是这个打分是第一次的打分
        最终得到了T个打分，即维度为(T, T)的矩阵
        '''
        scores = torch.matmul(q / (d_k ** 0.5), k.transpose(2, 3)) #(N, n_head, T, T)
        if mask is not None:
            # print(mask.unsqueeze(0).unsqueeze(0).shape, scores.shape)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0)==0, -1e9)
        scores = torch.nn.Softmax(dim=-1)(scores) #(N, n_head, T, T)
        # print(scores.shape, scores[2, 1, 0, :].sum())
        # print(scores[0, 0])

        '''
        (N, n_head, T, T)和(N, n_head, T, out_dim)做矩阵乘法，得到(N, n_head, T, out_dim)
        这个操作，可以拆分一下进行理解，首先我们忽略(N, n_head)两个维度，即变成(T, T)*(T, out_dim)
        我们取(T, T)的第零行，然后维度变成(1, T)*(T, out_dim)
        这个乘法的含义就很明确了，是第零个打分作用到所有的v，一共是T个v，打分中也有T个分数（和为1）。然后沿着T求和，即得到了(1, out_dim)
        然后取出第一行，也是一样的
        最终得到了T个结果，即维度为(T, out_dim)的矩阵
        '''
        output = torch.matmul(scores, v) #(N, n_head, T, out_dim)
        # print(output.shape)
        return output, scores

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_head, in_dim, out_dim): #实际上in_dim==out_dim，有利于残差的加法
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.out_dim = out_dim

        self.linear_q = torch.nn.Linear(in_features=in_dim, out_features=n_head*out_dim)
        self.linear_k = torch.nn.Linear(in_features=in_dim, out_features=n_head*out_dim)
        self.linear_v = torch.nn.Linear(in_features=in_dim, out_features=n_head*out_dim)

        self.scaled_dot_production_attention = ScaledDotProductAttention()
        self.linear = torch.nn.Linear(in_features=n_head*out_dim, out_features=out_dim) #这里out_features可以随意指定，这个就是encoder最终输出的qkv的维度，为了简便和out_dim一致

    def forward(self, q, k, v, mask=None):
        '''
        mask用来屏蔽一些特征，屏蔽的方式就是给scores置零之后再做softmax
        编码时：
            论文中并没有在编码时使用mask。
            这里mask可以用来屏蔽较pad的部分，其实不屏蔽也无所谓，网络可以学习到pad字符的含义，因此编码时mask=None
        解码时：
            最先输入到解码器的并不是编码器的输出，而是目标序列（训练是是目标序列，模型使用时是模型已输出的序列）。
            显然在训练时，我们知道完整的目标序列，但是当解码第5个输出时，需要屏蔽位于5之后的序列。
            然后解码器利用经过mask屏蔽的序列后，得到一个q，在结合编码器输出的k和v（k和v相同），一起进行后续的解码。
            mask屏蔽的方式可以是作用在scores上，在scores过softmax之前，给需要屏蔽的score赋值为一个很大的负数，然后经过softmask后就是0了。
        '''

        batch_size, len_q, len_kv = q.shape[0], q.shape[1], k.shape[1] #k和v的长度一直一致，但是在解码中，会出现q和kv长度不同的情况

        '''
        根据论文，经过全连接前后都叫QKV
        多个head合并在一起，这里用view分出真实维度
        '''
        q = self.linear_q(q).view(batch_size, len_q, self.n_head, self.out_dim) #(N, T, in_dim) --> (N, T, n_head * out_dim) --> (N, T, n_head, out_dim)
        k = self.linear_k(k).view(batch_size, len_kv, self.n_head, self.out_dim)
        v = self.linear_v(v).view(batch_size, len_kv, self.n_head, self.out_dim)


        q = q.transpose(1, 2) #(N, T, n_head, out_dim) --> (N, n_head, T, out_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # print(q.shape, k.shape, v.shape)

        output, scores = self.scaled_dot_production_attention(q, k, v, mask=mask)
        '''
        #(N, n_head, T, out_dim) --> (N, T, n_head, out_dim) --> (N, T, n_head * out_dim)
        这个操作相当于是论文中的concat，由于论文中的多个ScaledDotProductAttention在代码中用一个来实现了，因此view就相当于是concat了。
        这样做并发性好。
        '''
        output = output.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        # print(output.shape)

        output = self.linear(output) #(N, T, n_head * out_dim) --> (N, T, out_dim)
        # print(output.shape)

        return output, scores

class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.linear_1 = torch.nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.linear_2 = torch.nn.Linear(in_features=hidden_dim, out_features=in_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = torch.nn.ReLU()(x)
        x = self.linear_2(x)
        return x

class Encoder(torch.nn.Module):
    def __init__(self, n_head, in_dim, out_dim):
        super(Encoder, self).__init__()

        self.position_enc = PositionalEncoding(in_dim, out_dim)

        self.multi_head_attention_1 = MultiHeadAttention(n_head=n_head, in_dim=out_dim, out_dim=out_dim)
        self.layer_norm_1_1 = torch.nn.LayerNorm(out_dim)

        self.position_wise_feed_forward_1 = PositionWiseFeedForward(out_dim, hidden_dim=128)
        self.layer_norm_1_2 = torch.nn.LayerNorm(out_dim)

        self.scores_for_paint = None

    def forward(self, x):
        qkv = self.position_enc(x) #(N, T, 37) --> (N, T, 64) 在encoder中qkv三个tensor是一样的

        '''
        一下6行有效代码就是一个完整的encoder。
        在论文中，有n个encoder首尾相连，这里作为demo只用一个
        '''
        residual = qkv #根据论文，残差连接的是q。在encoder中，qkv是相同的；在decoder中，出现了k和v相同，但是和q不同的情形
        outputs, scores = self.multi_head_attention_1(qkv, qkv, qkv) #返回的scores用于可视化
        self.scores_for_paint = scores.detach().cpu().numpy() #用于绘制
        outputs = self.layer_norm_1_1(outputs + residual) #轮文中的Add & Norm
        # print(outputs.shape)

        residual = outputs
        outputs = self.position_wise_feed_forward_1(outputs)
        outputs = self.layer_norm_1_2(outputs + residual) #轮文中的Add & Norm
        # print(outputs.shape)

        return outputs

def get_subsequent_mask(seq):
    seq_len = seq.shape[1]
    ones = torch.ones((seq_len, seq_len), dtype=torch.int, device=seq.device)
    mask = 1 - torch.triu(ones, diagonal=1)
    # print(mask)
    return mask

class Decoder(torch.nn.Module):
    def __init__(self, n_head, in_dim, out_dim):
        super(Decoder, self).__init__()

        self.position_enc = PositionalEncoding(in_dim, out_dim)

        self.multi_head_attention_1_1 = MultiHeadAttention(n_head=n_head, in_dim=out_dim, out_dim=out_dim)
        self.layer_norm_1_1 = torch.nn.LayerNorm(out_dim)

        self.multi_head_attention_1_2 = MultiHeadAttention(n_head=n_head, in_dim=out_dim, out_dim=out_dim)
        self.layer_norm_1_2 = torch.nn.LayerNorm(out_dim)

        self.position_wise_feed_forward_1 = PositionWiseFeedForward(out_dim, hidden_dim=128)
        self.layer_norm_1_3 = torch.nn.LayerNorm(out_dim)

        self.scores_for_paint = None

    def forward(self, enc_outputs, target):
        qkv = self.position_enc(target) #在encoder中qkv三个tensor是一样的

        residual = qkv #根据论文，残差连接的是q。在encoder中，qkv是相同的；在decoder中，出现了k和v相同，但是和q不同的情形
        outputs, scores = self.multi_head_attention_1_1(qkv, qkv, qkv, mask=get_subsequent_mask(target)) #返回的scores用于可视化
        outputs = self.layer_norm_1_1(outputs + residual) #轮文中的Add & Norm，这里output就是q
        # print(outputs.shape)

        residual = outputs
        outputs, scores = self.multi_head_attention_1_2(outputs, enc_outputs, enc_outputs)
        self.scores_for_paint = scores.detach().cpu().numpy() #解码时每次会多次赋值，用最后的
        outputs = self.layer_norm_1_2(outputs + residual)

        residual = outputs
        outputs = self.position_wise_feed_forward_1(outputs)
        outputs = self.layer_norm_1_3(outputs + residual) #轮文中的Add & Norm
        # print(outputs.shape)

        return outputs

class Transformer(torch.nn.Module):
    def __init__(self, n_head):
        super(Transformer, self).__init__()

        self.encoder = Encoder(n_head, in_dim=37, out_dim=64) #37是human readable日期格式的字符串vocabulary数，和onehot向量长度一直
        self.decoder = Decoder(n_head, in_dim=12, out_dim=64) #12是machine readable日期格式的字符串vocabulary数，和onehot向量长度一直
        self.linear = torch.nn.Linear(in_features=64, out_features=12)

    def forward(self, x, y):
        enc_outputs = self.encoder(x)
        outputs = self.decoder(enc_outputs, y)
        outputs = self.linear(outputs)
        # print(outputs.shape)
        outputs = torch.nn.Softmax(dim=-1)(outputs)
        # print(outputs.shape)
        return outputs

    def size(self):
        size = sum([p.numel() for p in self.parameters()])
        print('%.2fKB' % (size * 4 / 1024))

if __name__ == '__main__':
    model = Transformer(n_head=4)
    model.size()

    batch_x = torch.randn(16, 10, 37) #(N, T, in_dim)
    batch_y = torch.randn(16, 7, 12) #(N, T, in_dim)
    # print(batch_x.shape)

    pred = model(batch_x, batch_y)
    print(pred.shape)
    print(pred[0][0])