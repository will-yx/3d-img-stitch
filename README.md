Gridwise stitching of 3D images
-----

1 - Overlap-aware tile registration

2 - Global optimization of image location

3 - On-the-fly downsample and stitching

Installation
-----
git clone the package

navigate to the parent directory

create a virtual python 3.9 environment with conda or mamba

	conda create -n 3dstitch python=3.9
  
activate the virtual environment

	conda activate 3dstitch

install the package

	pip install -e 3d-img-stitch

Usage
-----
in the 3d-img-stitch directory

 	python 3dstitch.py -i "input directory" -b True -o 0.1 -ch 3 -r 0 -ds 1 0.5 0.5
		
Parameters
-----
`--indir` or `-i` **str** : input directory (folder containing the images; or for batch processing, a folder of folders)

`--batch` or `-b` **bool** : True/False for batch processing; if True will look for subfolders that contain images and iterate through stitching each image set

`--overlap` or `-o` **float** : fraction of the x, y dimensions that overlaps with adjascent tiles

`--channel` or `-ch` **int** : number of channels to process

`--ref_channel` or `-r` **int** : reference channel to use for registration

`--downsample` or `-ds` **list of floats** : downscale factors for Z, Y, X dimensions *speeds up stitching and uses less RAM*
		
**Each image file is expected to be a single channel image stack containing all Z-slices**
Default filename structure used by the Miltenyi Blaze Ultramicroscope

	...[datestamp]...[00 x 00]_C00.ome.tif   --- zero-indexed [row x col]_C channel
