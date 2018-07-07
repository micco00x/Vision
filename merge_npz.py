import argparse
import numpy as np

if __name__ == "__main__":
    # Parse arguments from command line:
    parser = argparse.ArgumentParser(description="Mix two .npz datasets of shape (N, T, ...)")
    parser.add_argument("--ds1", required=True, help="First .npz file")
    parser.add_argument("--ds2", required=True, help="Second .npz file")
    parser.add_argument("--out", default="mix.npz", help="Path of generated .npz file (default=mix.npz)")
    args = parser.parse_args()

    # Load tensors (ds1):
    npzfile = np.load(args.ds1)
    ds1_videos = npzfile["videos"]
    ds1_classes = npzfile["videos_y"]

    # Load tensors (ds2):
    npzfile = np.load(args.ds2)
    ds2_videos = npzfile["videos"]
    ds2_classes = npzfile["videos_y"]

    # Check that the datasets are consistent:
    assert ds1_videos.shape[0] == ds2_videos.shape[0]
    assert ds1_videos.shape[1] == ds2_videos.shape[1]
    assert ds1_classes.shape[0] == ds2_classes.shape[0]
    for c_ds1, c_ds2 in zip(ds1_classes, ds2_classes):
        assert c_ds1 == c_ds2

    # Resize the np.array(s) and merge them:
    num_videos = ds1_videos.shape[0]
    num_frames = ds1_videos.shape[1]
    ds1_videos = np.reshape(ds1_videos, [num_videos, num_frames, np.prod(ds1_videos.shape[2:])])
    ds2_videos = np.reshape(ds2_videos, [num_videos, num_frames, np.prod(ds2_videos.shape[2:])])
    merged_ds = np.concatenate((ds1_videos, ds2_videos), axis=2)

    print("ds1_videos.shape:", ds1_videos.shape)
    print("ds2_videos.shape:", ds2_videos.shape)
    print("ds1_classes.shape:", ds1_classes.shape)
    print("ds2_classes.shape:", ds2_classes.shape)
    print("merged_ds.shape:", merged_ds.shape)

    # Save the new dataset:
    np.savez(args.out, videos=merged_ds, videos_y=ds1_classes)
