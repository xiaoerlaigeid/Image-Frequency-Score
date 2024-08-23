# Frequecny Metric for Image Generation quality


The proposed metric is based on frequency features statistics. 

## Requirements:
The provided source-code runs in python. The required packages can be installed using `conda` using the following:

```
conda create -n public_metric python=3.8
conda activate public_metric
conda install -c anaconda numpy scipy imageio matplotlib scikit-learn pandas
```

Optionally, for generating KDE cosine plots, `seaborn` can be installed:
```
conda install -c anaconda seaborn
```
The code has been tested with the packages described in `requirements.txt`.

## Running the software
The spatial metric can be computed by running the following command:
```
python spatial_metric.py --num_images 1000 --path_to_reals <path-to-folder-containing-real-pngs> --path_to_fakes <path-to-folder-containing-fake-pngs> --results_dir ./public_metric/
```


The frequency metric can be computed by running the following command:
```
python freq_metric.py --num_images 1000 --path_to_reals <path-to-folder-containing-real-pngs> --path_to_fakes <path-to-folder-containing-fake-pngs> --results_dir ./public_metric/
```




## Terms of use
The terms of use for this software are governed by the contents of the accompanying `LICENSE` file.



## Citation
Zhikai Yang, et al. "Efficient Generation of Synthetic Breast CT Slices By Combining Generative and Super-Resolution Models" MICCAI DeepBreast Workshop (2024)

Deshpande, Rucha, et al. "Report on the AAPM Grand Challenge on deep generative modeling for learning medical image statistics." ArXiv (2024).
