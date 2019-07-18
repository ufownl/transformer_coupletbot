import sys
import re
import math
import argparse
import http.server
import urllib.parse
import mxnet as mx
from vocab import Vocabulary
from couplet_seq2seq import CoupletSeq2seq
from transformer_utils import padding_mask

parser = argparse.ArgumentParser(description="Start a test http server.")
parser.add_argument("--beam_size", help="set the size of beam (default: 10)", type=int, default=10)
parser.add_argument("--addr", help="set address of coupletbot server (default: 0.0.0.0)", type=str, default="0.0.0.0")
parser.add_argument("--port", help="set port of coupletbot server (default: 80)", type=int, default=80)
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

print("Done.", flush=True)


class ChatbotHandler(http.server.BaseHTTPRequestHandler):
    _path_pattern = re.compile("^(/[^?\s]*)(\?\S*)?$")
    _param_pattern = re.compile("^([A-Za-z0-9_]+)=(.*)$")

    def do_GET(self):
        self._handle_request()
        sys.stdout.flush()
        sys.stderr.flush()

    def do_POST(self):
        self.do_GET()

    def _handle_request(self):
        m = self._path_pattern.match(self.path)
        if not m or m.group(0) != self.path:
            self.send_response(http.HTTPStatus.BAD_REQUEST)
            self.end_headers()
            return

        if m.group(1) == "/coupletbot/say":
            params = {}
            if m.group(2):
                for param in urllib.parse.unquote(m.group(2)[1:]).split("&"):
                    kv = self._param_pattern.match(param)
                    if kv:
                        params[kv.group(1)] = kv.group(2)

            content = params["content"]
            if not content:
                self.send_response(http.HTTPStatus.BAD_REQUEST)
                self.end_headers()
                return

            print(args.device_id, "say:", content)
            source = [vocab.char2idx(ch) for ch in content]
            print(args.device_id, "tokenize:", source)
            source = mx.nd.array(source, ctx=context).reshape((1, -1))
            src_len = mx.nd.array([source.shape[1]], ctx=context)
            enc_out, enc_self_attn = model.encode(source, src_len)
            sequences = [([vocab.char2idx("<GO>")], 0.0)]
            while True:
                candidates = []
                for seq, score in sequences:
                    if seq[-1] == vocab.char2idx("<EOS>") or len(seq) >= source.shape[1] + 2:
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

            reply = ""
            for token in sequences[0][0][1:-1]:
                reply += vocab.idx2char(token)

            print(args.device_id, "reply:", reply)

            self.send_response(http.HTTPStatus.OK)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET,POST")
            self.send_header("Access-Control-Allow-Headers", "Keep-Alive,User-Agent,Authorization,Content-Type")
            self.end_headers()
            self.wfile.write(reply.encode())
        else:
            self.send_response(http.HTTPStatus.NOT_FOUND)
            self.end_headers()
            return


httpd = http.server.HTTPServer((args.addr, args.port), ChatbotHandler)
httpd.serve_forever()
