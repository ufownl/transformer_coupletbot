import os
import time
import math
import random
import argparse
import mxnet as mx
from vocab import Vocabulary
from dataset import load_conversations, dataset_filter, make_vocab, tokenize, buckets, batches
from couplet_seq2seq import CoupletSeq2seq


def train(max_epochs, learning_rate, batch_size, sequence_length, sgd, context):
    print("Loading dataset...", flush=True)
    dataset = dataset_filter(load_conversations("data/couplets.conv"), sequence_length)
    if os.path.isfile("model/vocabulary.json"):
        vocab = Vocabulary()
        vocab.load("model/vocabulary.json")
    else:
        vocab = make_vocab(dataset)
        vocab.save("model/vocabulary.json")
    dataset = tokenize(dataset, vocab)
    split = len(dataset) - len(dataset) // 10
    training_set = dataset[:split]
    validating_set = dataset[split:]
    print("Training set:", len(training_set))
    print("Validating set:", len(validating_set))

    model = CoupletSeq2seq(vocab.size(), sequence_length)
    loss = mx.gluon.loss.SoftmaxCrossEntropyLoss(axis=2)

    if os.path.isfile("model/couplet_seq2seq.params"):
        model.load_parameters("model/couplet_seq2seq.params", ctx=context)
    else:
        model.initialize(mx.init.Xavier(), ctx=context)

    print("Learning rate:", learning_rate)
    if sgd:
        print("Optimizer: SGD")
        trainer = mx.gluon.Trainer(model.collect_params(), "SGD", {
            "learning_rate": learning_rate,
            "momentum": 0.5,
            "clip_gradient": 5.0
        })
    else:
        print("Optimizer: Adam")
        trainer = mx.gluon.Trainer(model.collect_params(), "Adam", {
            "learning_rate": learning_rate,
            "clip_gradient": 5.0
        })
    if os.path.isfile("model/couplet_seq2seq.state"):
        trainer.load_states("model/couplet_seq2seq.state")

    print("Training...", flush=True)
    for epoch in range(max_epochs):
        ts = time.time()

        random.shuffle(training_set)
        training_total_L = 0.0
        training_batches = 0
        for bucket, seq_len in buckets(training_set, [2 ** (i + 1) for i in range(int(math.log(sequence_length, 2)))]):
            for source, src_len, target, tgt_len, label in batches(bucket, vocab, batch_size, seq_len, context):
                training_batches += 1
                with mx.autograd.record():
                    output, enc_self_attn, dec_self_attn, context_attn = model(source, src_len, target, tgt_len)
                    L = loss(output, label)
                    L.backward()
                trainer.step(source.shape[0])
                training_batch_L = mx.nd.mean(L).asscalar()
                if training_batch_L != training_batch_L:
                    raise ValueError()
                training_total_L += training_batch_L
                print("[Epoch %d  Bucket %d  Batch %d]  batch_loss %.10f  average_loss %.10f  elapsed %.2fs" % (
                    epoch, seq_len, training_batches, training_batch_L, training_total_L / training_batches, time.time() - ts
                ), flush=True)
        training_avg_L = training_total_L / training_batches

        validating_total_L = 0.0
        validating_batches = 0
        ppl = mx.metric.Perplexity(ignore_label=None)
        for bucket, seq_len in buckets(validating_set, [2 ** (i + 1) for i in range(int(math.log(sequence_length, 2)))]):
            for source, src_len, target, tgt_len, label in batches(bucket, vocab, batch_size, seq_len, context):
                validating_batches += 1
                output, enc_self_attn, dec_self_attn, context_attn = model(source, src_len, target, tgt_len)
                L = loss(output, label)
                validating_batch_L = mx.nd.mean(L).asscalar()
                if validating_batch_L != validating_batch_L:
                    raise ValueError()
                validating_total_L += validating_batch_L
                probs = mx.nd.softmax(output, axis=2)
                ppl.update([label.reshape((-1,))], [probs.reshape((-1, vocab.size()))])
        validating_avg_L = validating_total_L / validating_batches

        print("[Epoch %d]  training_loss %.10f  validating_loss %.10f  %s %f  duration %.2fs" % (
            epoch + 1, training_avg_L, validating_avg_L, ppl.get()[0], ppl.get()[1], time.time() - ts
        ), flush=True)

        model.save_parameters("model/couplet_seq2seq.params")
        trainer.save_states("model/couplet_seq2seq.state")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a transformer_coupletbot trainer.")
    parser.add_argument("--max_epochs", help="set the max epochs (default: 100)", type=int, default=100)
    parser.add_argument("--learning_rate", help="set the learning rate (default: 1e-4)", type=float, default=1e-4)
    parser.add_argument("--batch_size", help="set the batch size (default: 128)", type=int, default=128)
    parser.add_argument("--sgd", help="using sgd optimizer", action="store_true")
    parser.add_argument("--device_id", help="select device that the model using (default: 0)", type=int, default=0)
    parser.add_argument("--gpu", help="using gpu acceleration", action="store_true")
    args = parser.parse_args()
    
    if args.gpu:
        context = mx.gpu(args.device_id)
    else:
        context = mx.cpu(args.device_id)

    train(args.max_epochs, args.learning_rate, args.batch_size, 32, args.sgd, context)
