import math
import argparse
import mxnet as mx
from vocab import Vocabulary
from dataset import pad_sentence
from couplet_seq2seq import CoupletSeq2seq
from transformer_utils import padding_mask

parser = argparse.ArgumentParser(description="Start a transformer_coupletbot tester.")
parser.add_argument("--beam", help="using beam search", action="store_true")
parser.add_argument("--beam_size", help="set the size of beam (default: 10)", type=int, default=10)
parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
args = parser.parse_args()

if args.gpu:
    context = mx.gpu(args.device_id)
else:
    context = mx.cpu(args.device_id)
beam_size = args.beam_size
sequence_length = 32

print("Loading vocabulary...", flush=True)
vocab = Vocabulary()
vocab.load("model/vocabulary.json")

print("Loading model...", flush=True)
model = CoupletSeq2seq(vocab.size(), sequence_length)
model.load_parameters("model/couplet_seq2seq.params", ctx=context)

while True:
    try:
        source = input("> ")
    except EOFError:
        print("")
        break
    source = [vocab.char2idx(ch) for ch in source]
    src_len = len(source)
    source = pad_sentence(source, vocab, [2 ** (i + 1) for i in range(int(math.log(sequence_length, 2)))])
    print(source)
    source = mx.nd.array(source, ctx=context).reshape((1, -1))
    src_len = mx.nd.array([src_len], ctx=context)
    enc_out, enc_self_attn = model.encode(source, src_len)

    if args.beam:
        sequences = [([vocab.char2idx("<GO>")], 0.0)]
        while True:
            candidates = []
            for seq, score in sequences:
                if seq[-1] == vocab.char2idx("<EOS>"):
                    candidates.append((seq, score))
                else:
                    target = mx.nd.array(seq, ctx=context).reshape((1, -1))
                    tgt_len = mx.nd.array([len(seq)], ctx=context)
                    context_attn_mask = padding_mask(target, source)
                    output, dec_self_attn, context_attn = model.decode(target, tgt_len, enc_out, context_attn_mask)
                    probs = mx.nd.softmax(output, axis=2)
                    beam = probs[0, -1].topk(k=beam_size, ret_typ="both")
                    for i in range(beam_size):
                        candidates.append((seq + [int(beam[1][i].asscalar())], score + math.log(beam[0][i].asscalar())))
            if len(candidates) <= len(sequences):
                break;
            sequences = sorted(candidates, key=lambda tup: tup[1], reverse=True)[:beam_size]

        scores = mx.nd.array([score for _, score in sequences], ctx=context)
        probs = mx.nd.softmax(scores)

        for i, (seq, score) in enumerate(sequences):
            text = ""
            for token in seq[1:-1]:
                text += vocab.idx2char(token)
            print(text, score, probs[i].asscalar())
            print(seq)
    else:
        sequence = [vocab.char2idx("<GO>")]
        target = mx.nd.array(sequence, ctx=context).reshape((1, -1))
        tgt_len = mx.nd.array([len(sequence)], ctx=context)
        while True:
            context_attn_mask = padding_mask(target, source)
            output, dec_self_attn, context_attn = model.decode(target, tgt_len, enc_out, context_attn_mask)
            index = mx.nd.argmax(output, axis=2)
            char_token = index[0, -1].asscalar()
            sequence += [char_token]
            if char_token == vocab.char2idx("<EOS>"):
                break;
            target = mx.nd.array(sequence, ctx=context).reshape((1, -1))
            tgt_len = mx.nd.array([len(sequence)], ctx=context)
            print(vocab.idx2char(char_token), end="", flush=True)
        print("") 
        print(sequence)
