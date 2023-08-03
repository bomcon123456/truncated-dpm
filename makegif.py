import glob
from PIL import Image
from natsort import natsorted

def make_gif(frame_folder):
    frames = [Image.open(image) for image in natsorted(list(glob.glob(f"{frame_folder}/xgen*")))]
    frame_one = frames[0]
    frame_one.save("out.gif", format="GIF", append_images=frames,
               save_all=True, duration=300, loop=1)
if __name__=="__main__":
    make_gif("./tmp")
