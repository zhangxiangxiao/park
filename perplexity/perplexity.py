"""Compute perplexity for a set of models."""

import argparse
import subprocess
import logging


PERPLEXITY = 'local/bin/perplexity -ngl 1000000 -m {model_file} -f {data_file}'
RESULT = 'perplexity/{data_name}_{model_name}_{quant}.txt'

# Model names and files.
MODELS = {
    'llama-7b': 'models/llama/LLaMa-7B-GGML/llama-7b.ggmlv3.{quant}.bin',
    'llama-13b': 'models/llama/LLaMa-13B-GGML/llama-13b.ggmlv3.{quant}.bin',
    'llama-30b': 'models/llama/LLaMa-30B-GGML/llama-30b.ggmlv3.{quant}.bin',
    'alpaca-lora-30b': 'models/alpaca/Alpaca-Lora-30B-GGML/Alpaca-Lora-30B.ggmlv3.{quant}.bin',
    'llama-2-7b': 'models/llama-2/Llama-2-7B-GGML/llama-2-7b.ggmlv3.{quant}.bin',
    'llama-2-7b-chat': 'models/llama-2/Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.{quant}.bin',
    'llama-2-13b': 'models/llama-2/Llama-2-13B-GGML/llama-2-13b.ggmlv3.{quant}.bin',
    'llama-2-13b-chat': 'models/llama-2/Llama-2-13B-chat-GGML/llama-2-13b-chat.ggmlv3.{quant}.bin',
}

# List of quantization levels.
QUANTS = ['q4_0', 'q4_1', 'q5_0', 'q5_1', 'q8_0']


# Put this here so readers can see the command-line arguments first :)
def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name', dest='data_name', default='wikitext-2',
        help='Dataset name.')
    parser.add_argument(
        '--data_file', dest='data_file', default='data/wikitext/wikitext-2-raw/'
        'wiki.test.raw', help='Dataset file.')
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
                model_file=model_file, data_file=args.data_file)
            logging.info('Executing %s.', perplexity)
            process = subprocess.run(
                perplexity, shell=True, capture_output=True)
            result = RESULT.format(
                data_name=args.data_name, model_name=model_name, quant=quant)
            logging.info('Write result to %s.', result)
            result_fd = open(result, 'wb')
            result_fd.write(process.stdout)
            result_fd.close()


if __name__ == '__main__':
    main()
