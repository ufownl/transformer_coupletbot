import mxnet as mx
from vocab import Vocabulary


def load_conversations(path):
    with open(path, "r") as f:
        raw = f.read()

    dataset = []
    for conv in raw.split("E\n")[1:]:
        qa = conv.split("\n")
        if len(qa) >= 2:
            src = qa[0][2:]
            tgt = qa[1][2:]
            dataset.append(((src, len(src)), (tgt, len(tgt))))

    return dataset


def dataset_filter(dataset, sequence_length):
    return [(src, tgt) for src, tgt in dataset if src[1] <= sequence_length and tgt[1] <= sequence_length]


def make_vocab(dataset):
    chars = sorted(list(set([ch for conv in dataset for sent in conv for ch in sent[0]])))
    return Vocabulary(chars)


def tokenize(dataset, vocab):
    return [tuple(([vocab.char2idx(ch) for ch in sent[0]], sent[1]) for sent in conv) for conv in dataset]


def buckets(dataset, buckets):
    min_len = -1
    for max_len in buckets:
        bucket = [(src, tgt) for src, tgt in dataset if src[1] > min_len and src[1] <= max_len]
        min_len = max_len
        if len(bucket) > 0:
            yield bucket, max_len


def batches(dataset, vocab, batch_size, sequence_length, ctx):
    src, tgt = zip(*dataset)
    src, tgt = list(src), list(tgt)
    src_tok, src_len = zip(*src)
    src_tok, src_len = list(src_tok), list(src_len)
    tgt_tok, tgt_len = zip(*tgt)
    tgt_tok, tgt_len = list(tgt_tok), list(tgt_len)
    n = len(dataset) // batch_size
    if n * batch_size < len(dataset):
        n += 1
    for i in range(0, n):
        start = i * batch_size
        src_bat = mx.nd.array(_pad_batch(src_tok[start: start + batch_size], vocab, sequence_length), ctx=ctx)
        src_len_bat = mx.nd.array(src_len[start: start + batch_size], ctx=ctx)
        tgt_bat = mx.nd.array(_pad_batch(_add_sent_prefix(tgt_tok[start: start + batch_size], vocab), vocab, sequence_length + 1), ctx=ctx)
        tgt_len_bat = mx.nd.array(tgt_len[start: start + batch_size], ctx=ctx) + 1
        lbl_bat = mx.nd.array(_pad_batch(_add_sent_suffix(tgt_tok[start: start + batch_size], vocab), vocab, sequence_length + 1), ctx=ctx)
        yield src_bat, src_len_bat, tgt_bat, tgt_len_bat, lbl_bat


def _add_sent_prefix(batch, vocab):
    return [[vocab.char2idx("<GO>")] + sent for sent in batch]


def _add_sent_suffix(batch, vocab):
    return [sent + [vocab.char2idx("<EOS>")] for sent in batch]


def _pad_batch(batch, vocab, seq_len):
    return [sent + [vocab.char2idx("<PAD>")] * (seq_len - len(sent)) for sent in batch]


if __name__ == "__main__":
    dataset = load_conversations("data/couplets.conv")
    print("dataset size: ", len(dataset))
    dataset = dataset_filter(dataset, 32)
    print("filtered dataset size: ", len(dataset))
    print("dataset preview: ", dataset[:10])
    vocab = make_vocab(dataset)
    print("vocab size: ", vocab.size())
    dataset = tokenize(dataset, vocab)
    print("tokenize dataset preview: ", dataset[:10])
    print("buckets preview: ", [(len(bucket), seq_len) for bucket, seq_len in buckets(dataset, [2, 4, 8, 16, 32])])
    print("batch preview: ", next(batches(dataset, vocab, 4, 32, mx.cpu())))
