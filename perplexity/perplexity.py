"""Compute perplexity for a set of models."""

import argparse
import subprocess
import logging


PERPLEXITY = 'local/bin/perplexity -ngl {layers} -m {model_file} -f {data_file}'
DATA_FILE = 'data/wikitext/wikitext-2-raw/wiki.test.raw'
RESULT = 'perplexity/wikitext-2_{model_name}_{quant}.txt'

# Model location.
MODELS = {
    'llama-7b': 'models/llama/LLaMa-7B-GGML/llama-7b.ggmlv3.{quant}.bin',
    'llama-2-7b': 'models/llama-2/Llama-2-7B-GGML/llama-2-7b.ggmlv3.{quant}.bin',
}

# List of quantization methods.
QUANTS = ['q4_0', 'q4_1', 'q5_0', 'q5_1', 'q8_0']


# Put this here so readers can see the command-line arguments first :)
def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--layers', type=int, dest='layers', default=1000000,
        help='Maximum number of layers to put on GPU.')
    return parser


def main(argv=None):
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    args = build_args().parse_args(argv)
    for model_name in MODELS:
        for quant in QUANTS:
            model_file  = MODELS[model_name].format(quant=quant)
            perplexity = PERPLEXITY.format(
                layers=args.layers, model_file=model_file, data_file=DATA_FILE)
            logging.info('Executing %s.', perplexity)
            process = subprocess.run(
                perplexity, shell=True, capture_output=True)
            result = RESULT.format(model_name=model_name, quant=quant)
            logging.info('Write result to %s.', result)
            result_fd = open(result, 'w')
            result_fd.write(process.stdout)
            result_fd.close()


if __name__ == '__main__':
    main()
