import mxnet as mx
from transformer_utils import padding_mask, Encoder, Decoder


class CoupletSeq2seq(mx.gluon.nn.Block):
    def __init__(self, vocab_size, max_len, layers=6, dims=512, heads=8, ffn_dims=2048, dropout=0.2, **kwargs):
        super(CoupletSeq2seq, self).__init__(**kwargs)
        with self.name_scope():
            self._encoder = Encoder(vocab_size, max_len, layers, dims, heads, ffn_dims, dropout)
            self._decoder = Decoder(vocab_size, max_len, layers, dims, heads, ffn_dims, dropout)
            self._output = mx.gluon.nn.Dense(vocab_size, flatten=False)

    def forward(self, src_seq, src_len, tgt_seq, tgt_len):
       out, enc_self_attn = self.encode(src_seq, src_len) 
       context_attn_mask = padding_mask(tgt_seq, src_seq)
       out, dec_self_attn, context_attn = self.decode(tgt_seq, tgt_len, out, context_attn_mask)
       return out, enc_self_attn, dec_self_attn, context_attn

    def encode(self, seq, seq_len):
        return self._encoder(seq, seq_len)

    def decode(self, seq, seq_len, enc_out, context_attn_mask):
        y, self_attn, context_attn = self._decoder(seq, seq_len, enc_out, context_attn_mask)
        y = self._output(y)
        return y, self_attn, context_attn
