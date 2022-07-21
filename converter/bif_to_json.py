from re import M
from pgmpy.readwrite import BIFReader
import argparse
import json

parser = argparse.ArgumentParser(description='Convert a .bif file representing a Bayesian network into a .json file that can be loaded by rsgm')
parser.add_argument('file', metavar='f', type=str,
                    help='a .bif file')

args = parser.parse_args()

reader = BIFReader(args.file)

newdict = {}
for key,value in reader.get_values().items():
    newdict[key] = value.tolist()

bn = {"network": reader.get_network_name(),
      "variables": reader.get_variables(), 
      "cpts": newdict,
      "states": reader.get_states(),
      "parents": reader.get_parents()}

print(json.dumps(bn))