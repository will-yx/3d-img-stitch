import os, sys, glob, tqdm, gc
import numpy as np
from skimage import io, registration, util
from skimage.restoration import rolling_ball
from skimage.transform import rescale

import networkx as nx
from scipy.optimize import least_squares
from tifffile import imread, imwrite
import matplotlib.pyplot as plt

from skimage.registration import phase_cross_correlation
def estimate_translation(fixed_np, moving_np):
    shift, _, _ = phase_cross_correlation(fixed_np, moving_np, upsample_factor=1)
    return shift  # Convert z,y,x to x,y,z

def register_tiles(ROWS,COLS,OVERLAP,ref_channel):
    shifts = {}
    sizes = None
    for r in range(ROWS):
        for c in range(COLS):
            if c==0:
                print(f"loading row {r} col {c} {img_files[r*COLS+c]}")
                fixed_np = imread(img_files[r*COLS+c])[ref_channel]
            else:
                print(f'pre-loaded row {r} col {c}')
                fixed_np = moving_np.copy()
                
            sizes = sizes or fixed_np.shape

            x_overlap=int(fixed_np.shape[2]*OVERLAP)
            y_overlap=int(fixed_np.shape[1]*OVERLAP)
                
            if r + 1 < ROWS:  # vertical neighbor
                #print("aligning down-adjacent tile")
                moving_np = imread(img_files[(r+1)*COLS+c])[ref_channel]

                u_o = fixed_np[:,fixed_np.shape[1]-y_overlap:,:]
                d_o = moving_np[:,:y_overlap,:]
                #print("calculating shifts")
                shift = estimate_translation(u_o, d_o)
                shift[1] = shift[1]+fixed_np.shape[1]-y_overlap
                shifts[((r, c), (r+1, c))] = shift

            if c + 1 < COLS:  # horizontal neighbor
                #print("aligning right-adjacent tile")
                moving_np = imread(img_files[r*COLS+c+1])[ref_channel]
                
                l_o = fixed_np[:,:,fixed_np.shape[2]-x_overlap:]
                r_o = moving_np[:,:,:x_overlap]
                #print("calculating shifts")
                shift = estimate_translation(l_o, r_o)
                shift[2] = shift[2]+fixed_np.shape[2]-x_overlap
                shifts[((r, c), (r, c+1))] = shift
    return shifts, sizes

def build_position_graph(shifts):
    G = nx.Graph()
    for (a, b), shift in shifts.items():
        G.add_edge(a, b, shift=shift)
    return G

def optimize_positions(shifts):
    G = build_position_graph(shifts)
    tile_indices = {(r, c): i for i, (r, c) in enumerate(sorted(G.nodes))}
    num_tiles = len(tile_indices)
    pos0 = np.zeros((num_tiles, 3))  # initial positions

    def residuals(flat_pos):
        pos = flat_pos.reshape((-1, 3))
        res = []
        for (a, b), shift in shifts.items():
            ia, ib = tile_indices[a], tile_indices[b]
            pred = pos[ib] - pos[ia]
            res.append(pred - shift)
        return np.concatenate(res)

    result = least_squares(residuals, pos0.ravel(), verbose=2)
    final_pos = result.x.reshape((-1, 3))

    global_positions = {tile: final_pos[tile_indices[tile]] for tile in tile_indices}
    return global_positions

def resample_to_canvas(global_positions, volume_shape, ch, ROWS, COLS, scale=False, ds=None, bksub=False):
    ds = np.array(ds)
    output_shape = np.array([0, 0, 0])
    min_coords = np.array([np.inf, np.inf, np.inf])

    for pos in global_positions.values():
        min_coords = np.minimum(min_coords, pos)

    shifts = {}
    for k, i in global_positions:
        shifts[(k,i)] = (np.floor(global_positions[k, i]-min_coords)*ds).astype(int)

    for pos in shifts.values():
        output_shape = np.maximum(output_shape, pos + np.ceil(volume_shape*ds)).astype(int)

    #ch_shape = np.array([n_ch, *output_shape]).astype(int)
    canvas = np.zeros(output_shape, dtype=np.int8)

    for (r, c), pos in tqdm.tqdm(shifts.items()):
        tile = imread(img_files[r*COLS+c])[ch]
        if scale:
            tile = rescale(tile, ds, anti_aliasing=False)
        if bksub and 0: # not working
            print('rolling ball subtraction')
            tile = tile - rolling_ball(tile, radius=bksub, num_threads=14)
        
        tile = util.img_as_ubyte(tile)
        z0, y0, x0 = pos.astype(int)
        zs, ys, xs = tile.shape
        canvas[z0:z0+zs, y0:y0+ys, x0:x0+xs] = tile

    s_ = np.zeros((ROWS, COLS))
    for s, pos in shifts.items():
        s_[s] = pos[0]
    zmin=np.max(s_)
    zmax=np.min(s_)+sizes[0]
    
    s_ = np.zeros((ROWS, COLS))
    for s, pos in shifts.items():
        s_[s] = pos[1]
    ymin = np.max(s_.astype(int), axis=1)[0]
    ymax = np.min(s_.astype(int), axis=1)[-1]
    
    s_ = np.zeros((ROWS, COLS))
    for s, pos in shifts.items():
        s_[s] = pos[2]
    xmin = np.max(s_.astype(int), axis=0)[0]
    xmax = np.min(s_.astype(int), axis=0)[-1]
    
    crop_min = np.array([zmin, ymin, xmin]).astype(int)
    crop_max = np.array([zmax, ymax, xmax]).astype(int)
    
    return canvas[crop_min[0]:crop_max[0]

def run_stitching(indir, batch=False, overlap=0.1, n_ch=3, ref_channel=0, ds=[1,1,1]):
    if batch = False:
        img_dirs = [indir]
    else:
        img_dirs = glob.glob(indir+'/*')
        print(f"found {len(img_dirs)} directories") 
    for img_dir in img_dirs:
        #get image files
        img_files=sorted(glob.glob(img_dir+f'/*]_C{use_ch:02d}.ome.tif'))
        
        #get shape of tiles ###likely to break... use better heuristics
        ROWS = int(img_files[-1][-20:-18])+1
        COLS = int(img_files[-1][-15:-13])+1

        print(f'found {ROWS} x {COLS} tiled images in {img_dir}')
        assert len(img_files) == ROWS*COLS, 'Wrong number of ome.tif files found!'
        
        print("Registering tile pairs...")
        shifts, sizes = register_tiles(ROWS,COLS,OVERLAP,ref_channel)
        print("Optimizing tile layout...")
        global_positions = optimize_positions(shifts)
        
        outdir = os.path.join(img_dir,'merged')
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        print(f"Output directory: {outdir}")
        
        for z in range(n_ch):
            print(f'Stitching CH{z}...')
            stitched = resample_to_canvas(global_positions, sizes, z, scale=True, ds=ds, bksub=None)
            
            #save stitched file
            print('Saving image file...')
            imwrite(f"{outdir}/stitched_py_ch{z}.tif", stitched)
            if not os.path.isfile(f"{outdir}/stitched_py_ch{z}.tif"):
                raise "file not written"
            else:
                print('Done')
                del stitched
                gc.collect()
                gc.collect()

import argparse

if __name__ == '__main__':
  __spec__ = None
  
  parser = argparse.ArgumentParser(
    prog='3D image stitcher',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent('''\
        Gridwise stitching of 3D images
        --------------------------------
            1 - Overlap-aware tile registration
            2 - Global optimization of image location
            3 - On-the-fly downsample and stitching
            
            Default filename structure used by the Miltenyi Blaze Ultramicroscope
                ...[datestamp]...[00 x 00]_C00.ome.tif   --- zero-indexed [row x col]_C channel
            use -b option for batch processing any number of image folders
        '''))
        
  parser.add_argument("-i", "--indir", nargs=1, help = "Input directory")
  parser.add_argument("-b", "--batch", help = "Batch process a directory of image directories", default = False)
  parser.add_argument("-o", "--overlap", help = "Amount of tile-tile overlap", default = 0.1)
  parser.add_argument("-ch", "--channels", help = "Number of channels", default = 1)
  parser.add_argument("-r", "--ref_channel", help = "Reference channel for alignment", default = 0)
  parser.add_argument("-ds", "--downsample", nargs=3, help = "Ratio of downsample in z y x", default = [1,1,1])
  
  args = parser.parse_args()
  
  run_stitching(args.indir, batch=args.batch, overlap=args.overlap, n_ch=args.channels, ref_channel=args.ref_channel, ds=args.downsample)