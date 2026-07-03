from datasets import build_dataset
import argparse
import polyscope as ps

def parse_args():
    parser = argparse.ArgumentParser(description="Run feature-based NN for a baseline assignment quality")
    parser.add_argument('-d', '--dataset', default='FAUST_r', help='dataset name - as in the data folder')
    return parser.parse_args()


def main():
    args = parse_args()
    print(args.dataset)


if __name__=="__main__":
    dataset_opt = {"name":"Faust_r", "type":"SingleFaustDataset"}
    ds = build_dataset(dataset_opt)
    item = ds.__getitem__(1)

    ps.init()
    ps.register_surface_mesh("name", item["verts"], item["faces"])

    if ds.flip_up:
        ps.set_up_dir("neg_y_up")
    ps.show()