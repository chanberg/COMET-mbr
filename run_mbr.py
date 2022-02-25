import argparse
import itertools

import numpy as np
from comet import download_model, load_from_checkpoint

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-t', '--transl_file', type=argparse.FileType('r'), required=True, help='path to file containing sampled support hypotheses (number of lines should be a multiple of the number of lines in src file).')
    ap.add_argument('-s', '--source_file', type=argparse.FileType('r'), required=True, help='path to file containing source sentences.')
    ap.add_argument('-c', '--candidate_file', type=argparse.FileType('r'), required=True, help='path to file containing sampled candidate sentences (number of lines should be a multiple of the number of lines in src file).')
    ap.add_argument('-o', '--output_file', type=argparse.FileType('w'), required=True, help='path to output file.')
    ap.add_argument('-ns', '--n_support', type=int, required=True, default=1, help='number of support samples per src sentence.')
    ap.add_argument('-nc', '--n_candidates', type=int, required=True, default=1, help='number of candidate samples per src sentence.')
    ap.add_argument('-b', '--batch_size', type=int, required=False, default=8, help='batch size for MBR decoding.')
    ap.add_argument('-g', '--gpus', type=int, required=False, default=1, help='how many GPUs to use, 0 == CPU.')
    ap.add_argument('-m', '--model_name', type=str, required=False, default='wmt20-comet-da', help='COMET model name.')
    return ap.parse_args()

def chunk(iterator, size):
    while True:
        chunk = list(itertools.islice(iterator, size))
        if chunk:
            yield chunk
        else:
            break

def main(args):

    model_path = download_model(args.model_name)
    model = load_from_checkpoint(model_path)

    data = []

    for source, candidates, support in zip(args.source_file, chunk(args.candidate_file, args.n_candidates), chunk(args.transl_file, args.n_support)):
        example = {}
        candidates = [c.strip() for c in candidates]
        support = [s.strip() for s in support]
        example['src'] = source.strip()
        example['mt'] = candidates
        example['ref'] = support
        data.append(example)

    batched_matrices = model.get_utility_scores(data, args.batch_size, gpus=args.gpus)

    for matrices, examples in zip(batched_matrices, chunk((x for x in data), args.batch_size)):
        if args.gpus > 0:
            matrices = matrices.cpu()
        matrices = np.reshape(matrices, [len(examples), len(examples[0]['mt']), -1])

        for matrix, example in zip(matrices, examples):
            mbr_scores = np.average(matrix, axis=-1)
            prediction_idx = np.argmax(mbr_scores, axis=-1)
            args.output_file.write(example['mt'][prediction_idx]+'\n')


if __name__ == '__main__':

    args = parse_args()
    main(args)
