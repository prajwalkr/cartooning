# Image Cartooning
![Cartooning Input-Output](https://drive.google.com/uc?export=view&id=1rmnUgRa6pzfkYkIOseZGbuToNNSz7xNm)
We build an application to cartoonify a given image using basic image processing techniques. We first implement the paper [Toonify: Cartoon Photo Effect Application](https://stacks.stanford.edu/file/druid:yt916dh6570/Dade_Toonify.pdf) and also propose our own improvements.

#### Key features implemented:
 - Not explicitly restricted to any specific image type. 
 - The cartoon effects match or are even better than popular online cartooning tools ([Link1](http://funny.pho.to/cartoon/), [Link2](https://www.cartoonize.net/))
- More detailed edges than shown in the paper.
 - Special handling of facial inputs to cartoonize faces of people. 
 - Highly customizable for every input image using command line args

### Setup with 3 simple steps
 - Get ``Python 3`` and ``pip``. Code has been tested with ``Python==3.6.8``
 - Create a virtual environment [(how to)](https://stackoverflow.com/a/23842752/4219775) and activate it. Using ``conda`` or system Python should also work, although the former has not been tested. 
 - Get the code from this repository. In the root folder of the repository, just run ``sh setup.sh``. Sit back and relax as some packages like ``dlib`` take time to install. 
 
 ### Let's Cartoonify!
 
The code takes in an input image path and outputs two cartoon images from the: (i) Original paper implementation (ii) Improved implementation

    python cartooning.py images/whale.jpg -s results/out_paper.jpg -o results/out.jpg

<p float="left">
  <img src="https://drive.google.com/uc?export=view&id=1ln67n4NpgzMpf4ivGk987wXuUTeTYH_Z" width="280" />
  <img src="https://drive.google.com/uc?export=view&id=1WQ43wzH5dEwTSqawJa4Bb4KwkiTc6TY9" width="280" /> 
  <img src="https://drive.google.com/uc?export=view&id=12eoDA01TyKt5CFEXlDsNAcrrudJhnRdN" width="280" />
</p>

 This will display two matplotlib windows showing the cartooned image for the paper and our implementation respectively. Both the output images are also saved in the paths specified above. 

Almost all the options for cartooning can be tuned in the command line. Explore the various options: 

    >> python cartooning.py -h
	

    usage: cartooning.py [-h] [-m MEDIAN_KERNEL_SIZE] [-f DILATION_SIZE]
                         [-q QUANTIZATION_FACTOR] [-C C] [-b BLOCKSIZE]
                         [-r REDUCE_SPECKLES] [-u UNEDGED] [-p PART]
                         [-c CLUSTER_SIZE] [-s OUTPUT_IMAGE_PAPER]
                         [-o OUTPUT_IMAGE]
                         [--overlay_parts [OVERLAY_PARTS [OVERLAY_PARTS ...]]]
                         input
    
        Cartoon any given image
        
        positional arguments:
          input                 Input image
        
        optional arguments:
          -h, --help            show this help message and exit
          -m MEDIAN_KERNEL_SIZE, --median_kernel_size MEDIAN_KERNEL_SIZE
                                Increase this value to reduce no. of edges & get a
                                more smooth output
          -f DILATION_SIZE, --dilation_size DILATION_SIZE
                                Increase this value to get thicker edges
          -q QUANTIZATION_FACTOR, --quantization_factor QUANTIZATION_FACTOR
                                Controls cartooning effect by creating blobs, increase
                                to get larger, unifrom blobs
          -C C, --C C           thresholding parameter C
          -b BLOCKSIZE, --blockSize BLOCKSIZE
                                thresholding parameter blockSize
          -r REDUCE_SPECKLES, --reduce_speckles REDUCE_SPECKLES
                                reduce speckle edges
          -u UNEDGED, --unedged UNEDGED
                                =True displays original implementation without edge
                                overlays
          -p PART, --part PART  Face Part to add face effects
          -c CLUSTER_SIZE, --cluster_size CLUSTER_SIZE
                                Decrease this to get more blobby cartoon-style faces
          -s OUTPUT_IMAGE_PAPER, --output_image_paper OUTPUT_IMAGE_PAPER
                                Path to save the output of the paper implementation
          -o OUTPUT_IMAGE, --output_image OUTPUT_IMAGE
                                Path to save the output of the improved implementation
          --overlay_parts [OVERLAY_PARTS [OVERLAY_PARTS ...]]
                                Draw lines using over face parts. Not necessary
                                currently.

Please feel free to go through the code. It has been commented at necessary places for ease of understanding.
