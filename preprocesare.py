from glob import glob
import shutil
import os

"""
OBS! ESTE NECESARA INCLUDEREA ACESTOR LIBRARII :) - se pot descarca cu pip ( ex. : pip install glob2 )

Creare grupuri dicom pentru images si labels
- primul script trebuie rulat pentru preprocesarea label-urilor imaginilor
- al doilea script trebuie rulat pentru preprocesarea imaginilior
"""
def preprocessLabels():
    
    in_path = 'C":/the_right_path/dicom_files/labels'
    out_path = 'C:/the_right_path/dicom_groups/labels'


# Loop pentru toti pacientii
    for patient in glob(in_path + '/*'):
        patient_name = os.path.basename(os.path.normpath(patient))  # normalizare daca este vreo greseala
        number_folders = int(len(glob(patient+'/*')) / 64)  # grup de cate 64
    
        for i in range(number_folders):
            output_path_name = os.path.join(out_path, f"{patient_name}_{i}")
            os.mkdir(output_path_name)        
        
            for j, file in enumerate(glob(patient+'/*')):
                if j == 64 + 1:  
                    break
                shutil.move(file, output_path_name)  # mut file-ul in directorul dorit
                
                
def preprocessImages():
    
    in_path = 'C:/the_right_path/dicom_files/images'
    out_path = 'C:/the_right_path/dicom_groups/images'


# Loop pentru toti pacientii
    for patient in glob(in_path + '/*'):
        patient_name = os.path.basename(os.path.normpath(patient))  # # normalizare daca este vreo greseala
        number_folders = int(len(glob(patient+'/*')) / 64)  # grup de cate 64
    
        for i in range(number_folders):
            output_path_name = os.path.join(out_path, f"{patient_name}_{i}")
            os.mkdir(output_path_name)        
        
            for j, file in enumerate(glob(patient+'/*')):
                if j == 64 + 1:  
                    break
                shutil.move(file, output_path_name)  # mut file-ul in directorul dorit
                
  
