import MaskExam as msk
import numpy as np
import argparse
import cv2
import os


parser = argparse.ArgumentParser(description='Apply masks to video using MaskRCNN')
parser.add_argument('--input', dest='input', type=str, help='input video to be processed', required=True)
parser.add_argument('--outfolder', dest='outfolder', type=str, default='./masked_videos', help='folder where to save the output video')
parser.add_argument('--output', dest='output', type=str, default=None, help='filename for the output video')
parser.add_argument('--startsecond', dest='startsecond', type=int, default=0, help='filename for the output video')
parser.add_argument('--endsecond', dest='endsecond', type=int, default=-1, help='filename for the output video')
args = parser.parse_args()

input_video = args.input
output_video = args.output
output_folder = args.outfolder
start_second = args.startsecond
end_second = args.endsecond

if output_video == None:
	input_name = os.path.splitext(os.path.basename(input_video))[0]
	output_video = 'masked_' + input_name + '.mp4'
	
output_video = os.path.join(output_folder, output_video)

if os.path.isfile(output_video):
	print("Output file already exists. Overwrite? Y,[n]")
	overwrite = input()
	if overwrite != 'Y':
		print("Computation aborted")
		quit()

# video capture for readinf video frames
vidcap = cv2.VideoCapture(input_video)

width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = vidcap.get(cv2.CAP_PROP_FPS)
total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

# video writer for writing new video with masks
#fourcc = cv2.VideoWriter_fourcc(*'MP4V')
video = cv2.VideoWriter(output_video, 0x00000021, fps, (width,height))

start_frame = int(start_second * fps)
end_frame = int(end_second * fps)

if end_frame < 0:
	end_frame = total_frames

# reading video frames
_, _ = vidcap.read() #first read to reach first frame
success = True

n_frame = 0
while success:    
	success, image = vidcap.read()
	
	if n_frame < start_frame or not success:
		continue
	
	mask_frame = msk.apply_masks(image)
	video.write(np.uint8(mask_frame))
	
	n_frame += 1
	
	print("Computing masked frames: {:.2f}%".format((n_frame-start_frame)/(end_frame-start_frame) * 100), end='\r')
	
	if n_frame == end_frame:
		break

cv2.destroyAllWindows()
vidcap.release()
video.release()

print("\nVideo creation completed\n")
