import tensorflow as tf
import argparse

if __name__ == "__main__":
    # Parse arguments from command line:
    parser = argparse.ArgumentParser(description="Convert tensorboard event tag to python list")
    parser.add_argument("--event", required=True, help="Path to tensorbord event")
    parser.add_argument("--tag", required=True, help="Tag of the event value")
    args = parser.parse_args()

    tag_value_list = []

    for e in tf.train.summary_iterator(args.event):
        for v in e.summary.value:
            if v.tag == args.tag:
                tag_value_list.append((e.step, v.simple_value))

    print(tag_value_list)
