import numpy as np 
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="check numpy array")
parser.add_argument("numpy_array", help="numpy array")
args = parser.parse_args()

numpy_array = np.load(args.numpy_array)
print(numpy_array)
