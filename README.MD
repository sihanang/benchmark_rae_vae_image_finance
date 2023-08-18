# Benchmarking Deterministic Autoencoders for Generative Modeling: Applications to Image/Vision and Finance
## This contains the code for the above MSc Dissertation

#### Written by Ang Si Han

## Setting Up
The code was written under the following setup:
* Using this version of Anaconda: https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh, followed by
* pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
* pip install opencv-python

## Navigation within folder structure
The msc_edinburgh_thesis_material_submission folder contains two folders:
* data
* scripts_and_model_related_files

#### To navigate around the data folder structure
```
msc_edinburgh_thesis_material_submission
    |____ data
            |___ finance_data
            |     |___csv files
            |            
            |___  image_data
		  |
		  |____FashionMNIST
		  |	|
		  |	|__raw
		  |	|   |___files
		  |	|
		  |	|__proc
		  |	    |___GMM
		  |	    |	
		  |	    |___rndmN
		  |	    |	
		  |	    |___test
		  |	    |	
		  |	    |___test_recon
		  |
		  |
		  |____stanford_cars
			|
			|__proc
			    |___gmm
			    |	
			    |___rndmN
			    |	
			    |___test
			    |	
			    |___test_recon

```
#### To navigate around the scripts_and_model_related_files folder structure
```
msc_edinburgh_thesis_material_submission
    |____ scripts_and_model_related_files 
            |___ scripts
            |     
            |            
            |___  model_files
		  |___cars_models__ .pth model files for cars dataset
		  |
		  |___finance models__ .pth model files for implied volatility surface data
		  |
		  |___csv files__ .pth model files for fashion MNIST dataset

```
#### Training Models
* Found scripts_and_model_related_files
* Model training scripts are named starting with the model name followed by the data set or domain
* For Fashion MNIST, the scripts for training are: RAE_fmnist_final.py, vae_fmnist.py, WAE_fmnist.py
* For Cars, the scripts for training are: rae_stanford_cars_final.py, rae_stanford_cars_spectral_norm.py, vae_stanford_cars.py, wae_stanford_cars.py
* For implied volatility surface, the scripts are: RAE_Finance.py, VAE_Finance.py, WAE_Finance.py
* Note that the scripts depend upon the losses.py which contains user defined classes for the different models and spectral_norm_layers.py for SN
* Cars models depends upon cars_data_gen.py and cars_preprocess.py.


#### Model Evaluation
* Found scripts_and_model_related_files
* Model Evaluation for images are found in compute_fid.py which depends on fid.py, model_repo.py and inception.py 
  as well as the pretrained model file pt_inception-2015-12-05-6726825d.pth. Cars models depends upon cars_data_gen.py and cars_preprocess.py.
* Model evaluation for finance dataset are mae_eval_complete_vol_surface.py, test_recon_mae_and_gridbased_completion.py and finance_gen_check.py
* Due to file size limitation, some of the files like pt_inception-2015-12-05-6726825d.pth are uploaded on the google drive: https://drive.google.com/drive/folders/1J2onGDemUzY0jZ3kD4JGXH6voUvAip4N?usp=sharing

#### Trained Model files
* Found in the respective subfolders in the model_files directory.
All materials could be found on: https://drive.google.com/drive/folders/1J2onGDemUzY0jZ3kD4JGXH6voUvAip4N?usp=sharing
