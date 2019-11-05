*Code is tested on python3.5, pytorch 1.0.1 cuda10.0.130_cudnn7.4.2.2, and jupyter 1.0.0*
# Dependencies
Packages needed to run is in `environment.yml`. Create a virtual environment to run this, (optionally rename the environment's name by tweaking the YML file). 

To create a virtual env and install required packages, please use **miniconda3**, and run:

```bash
conda env create -f environment.yml
```
# Data preparation
## Folder structure
    .# THIS IS $HOME dir
    ├── data                        # Contains annotations, images, and vocabularies
      ├── annotations               # Contains json files for test, train, val of a specific task
      ├── img                       # Contains images for test, train, val of a specific task
      ├── vocab                     # Contains vocabulary of a specific task
    ├── dataset                     # Documentation files (alternatively `doc`)
      ├── original 
        ├── annotations_trainval2014  # Contains json files from MSCOCO
        ├── train2014                 # Train images from MSCOCO
        ├── val2014                   # val images from MSCOCO
      ├── processed 
        ├── train                   # Contains 80 directories of 80 classes with training images
        ├── val                     # Contains 80 directories of 80 classes with validation images
        ├── test                    # Contains 80 directories of 80 classes with testing images
    ├── infer                       # Contains predictions of model on the test images of a specific task
      ├── json
    ├── models                      # Contains models for tasks after training
      ├── one                       # Contains models when adding a class
      ├── once                      # Contains models when multiple classes at once
      ├── seq                       # Contains models when multiple classes one by one
    ├── png                         # Some sample images for testing
    ├── prepro                      # Tools and utilities for processing data
    ├── LICENSE
    └── README.md
## Data processing
Download [MS-COCO 2014 dataset](http://cocodataset.org/#download) and put them into directories like above Folder structure.

First we read the original MS-COCO and classify (+resize) to 80 different classes to 80 different folders in `processed/`. In `prepro/`, run
```python3
python classify_and_resize.py
```
## coco-caption for evaluation
For evaluation of this project, we use `coco-caption` package from Liu's [repository](https://github.com/daqingliu/coco-caption) but remove some redundancies. Download our modified `coco-caption` [here](https://drive.google.com/open?id=1_aFAIRpo18yvqmSM8V_1uBkpk04ZzqbO).
# Training & inference
* Step 1: Create data for 2to21. In `prepro/`, run
```bash
bash process_data.sh 2to21 2to21
```

* Step 2: Train model 2to21 to get model for fine-tuning. From $HOME, run:
```python3
python train.py --task_type one --task_name 2to21
```
**However, it is recommended  to use our model and vocabulary for a better reproducibility (Random weight initilization and data shuffling shift the final results). Please download the model and vocabulary from [HERE](https://drive.google.com/open?id=1_aFAIRpo18yvqmSM8V_1uBkpk04ZzqbO). After that, put the vocabulary at `data/vocab/2to21/` (overwrite), and the model at `models/one/2to21/best/`.**

* Step 3: Infer captions to compute metrics by changing `infer` section in `config.yaml` file. Here we test this model on 2to21's test set:
```yaml
infer:
  img_path: 'data/img/2to21/test/'                               # Image to be tested
  json_path: 'data/annotations/2to21/captions_test.json'         # Annotations of images to be tested
  model: 'models/one/2to21/best/BEST_checkpoint_ms-coco.pth.tar' # Model to test
  vocab_path: 'data/vocab/2to21/vocab.pkl'                       # Vocab corresponding to the model
  prediction_path: 'infer/json/2to21_on_2to21/'                  # Test model 1 with fine-tuning on 2to21 test set
  id2class_path: 'dataset/processed/id2class.json'               # Skip it
```

then run:

```
python infer.py
```
* Step 4: Compute metrics by using `coco-caption` package provided. Run `coco-caption/cocoEvalCapDemo.ipynb` by jupyter notebook. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/2to21/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/2to21_on_2to21/prediction.json'
```
* Step 5: Generate a sentence for a picture. Run:
```python3
python sample.py --model YOUR_MODEL --image IMAGE_TO_INFER --vocab VOCAB_FOR_THE_MODEL
```

Example: 

```python3
python sample.py --model models/one/2to21/best/BEST_checkpoint_ms-coco.pth.tar --image png/cat2.jpg 
--vocab data/vocab/2to21/vocab.pkl
```
## Addition of one class 
### Fine-tuning
* Step 1: Create data for this task. In `prepro/`, run
```bash
bash process_data.sh 1 1
```
* Step 2: Train model. Fine-tune from model 2to21, then from $HOME run :
```python3
python train.py --task_type one --task_name 1 --fine_tuning
```
* Step 3: Infer captions to compute metrics by changing `infer` section in `config.yaml` file. Here we test this model on 2to21's test set:
```yaml
infer:
  img_path: 'data/img/2to21/test/'                               # Image to be tested
  json_path: 'data/annotations/2to21/captions_test.json'         # Annotations of images to be tested
  model: 'models/one/1/best/BEST_checkpoint_ms-coco.pth.tar'     # Model to test
  vocab_path: 'data/vocab/1/vocab.pkl'                           # Vocab corresponding to the model
  prediction_path: 'infer/json/1_on_2to21/'                      # Test model 1 with fine-tuning on 2to21 test set
  id2class_path: 'dataset/processed/id2class.json'               # Skip it
```

then run:

```
python infer.py
```
* Step 4: Compute metrics on the old task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/2to21/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/1_on_2to21/prediction.json'
```
* Step 5: Infer captions to compute metrics 1's test set:
```yaml
infer:
  img_path: 'data/img/1/test/'                                   # Image to be tested
  json_path: 'data/annotations/1/captions_test.json'             # Annotations of images to be tested
  model: 'models/one/1/best/BEST_checkpoint_ms-coco.pth.tar'     # Model to test
  vocab_path: 'data/vocab/1/vocab.pkl'                           # Vocab corresponding to the model
  prediction_path: 'infer/json/1_on_1/'                          # Test model 1 with fine-tuning on 2to21 test set
  id2class_path: 'dataset/processed/id2class.json'               # Skip it
```

then run:

```
python infer.py
```
* Step 6: Compute metrics on the new task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/1/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/1_on_1/prediction.json'
```

### Pseudo-labeling
* Step 2: Train model. Fine-tune from model 2to21, then from $HOME run :
```python3
python train.py --task_type one --task_name 1 --fine_tuning --lwf
```
* Step 3: Infer captions to compute metrics by changing `infer` section in `config.yaml` file. Here we test this model on 2to21's test set:
```yaml
infer:
  img_path: 'data/img/2to21/test/'                               # Image to be tested
  json_path: 'data/annotations/2to21/captions_test.json'         # Annotations of images to be tested
  model: 'models/one/1_lwf/best/BEST_checkpoint_ms-coco.pth.tar' # Model to test
  vocab_path: 'data/vocab/1/vocab.pkl'                           # Vocab corresponding to the model
  prediction_path: 'infer/json/1_on_2to21_lwf/'                  
  id2class_path: 'dataset/processed/id2class.json'               # Skip it
```

then run:

```
python infer.py
```
* Step 4: Compute metrics on the old task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/2to21/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/1_on_2to21_lwf/prediction.json'
```
* Step 5: Infer captions to compute metrics 1's test set:
```yaml
infer:
  img_path: 'data/img/1/test/'                                   # Image to be tested
  json_path: 'data/annotations/1/captions_test.json'             # Annotations of images to be tested
  model: 'models/one/1_lwf/best/BEST_checkpoint_ms-coco.pth.tar' # Model to test
  vocab_path: 'data/vocab/1/vocab.pkl'                           # Vocab corresponding to the model
  prediction_path: 'infer/json/1_on_1_lwf/'                          
  id2class_path: 'dataset/processed/id2class.json'               # Skip it
```

then run:

```
python infer.py
```
* Step 6: Compute metrics on the new task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/1/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/1_on_1_lwf/prediction.json'
```
### Freeze encoder
* Step 2: Train model. Fine-tune from model 2to21, then from $HOME run :
```python3
python train.py --task_type one --task_name 1 --fine_tuning --freeze_enc
```
* Step 3: Infer captions to compute metrics by changing `infer` section in `config.yaml` file. Here we test this model on 2to21's test set:
```yaml
infer:
  img_path: 'data/img/2to21/test/'                                      # Image to be tested
  json_path: 'data/annotations/2to21/captions_test.json'                # Annotations of images to be tested
  model: 'models/one/1_freeze_enc/best/BEST_checkpoint_ms-coco.pth.tar' # Model to test
  vocab_path: 'data/vocab/1/vocab.pkl'                                  # Vocab corresponding to the model
  prediction_path: 'infer/json/1_on_2to21_freeze_enc/'                  
  id2class_path: 'dataset/processed/id2class.json'                      # Skip it
```

then run:

```
python infer.py
```
* Step 4: Compute metrics on the old task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/2to21/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/1_on_2to21_freeze_enc/prediction.json'
```
* Step 5: Infer captions to compute metrics 1's test set:
```yaml
infer:
  img_path: 'data/img/1/test/'                                              # Image to be tested
  json_path: 'data/annotations/1/captions_test.json'                        # Annotations of images to be tested
  model: 'models/one/1_freeze_enc/best/BEST_checkpoint_ms-coco.pth.tar'     # Model to test
  vocab_path: 'data/vocab/1/vocab.pkl'                                      # Vocab corresponding to the model
  prediction_path: 'infer/json/1_on_1_freeze_enc/'                          
  id2class_path: 'dataset/processed/id2class.json'                          # Skip it
```

then run:

```
python infer.py
```
* Step 6: Compute metrics on the new task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/1/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/1_on_1_freeze_enc/prediction.json'
```

### Freeze decoder
* Step 2: Train model. Fine-tune from model 2to21, then from $HOME run :
```python3
python train.py --task_type one --task_name 1 --fine_tuning --freeze_dec
```
* Step 3: Infer captions to compute metrics by changing `infer` section in `config.yaml` file. Here we test this model on 2to21's test set:
```yaml
infer:
  img_path: 'data/img/2to21/test/'                               
  json_path: 'data/annotations/2to21/captions_test.json'        
  model: 'models/one/1_freeze_dec/best/BEST_checkpoint_ms-coco.pth.tar'
  vocab_path: 'data/vocab/1/vocab.pkl'                  
  prediction_path: 'infer/json/1_on_2to21_freeze_dec/'                  
  id2class_path: 'dataset/processed/id2class.json'                
```

then run:

```
python infer.py
```
* Step 4: Compute metrics on the old task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/2to21/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/1_on_2to21_freeze_dec/prediction.json'
```
* Step 5: Infer captions to compute metrics 1's test set:
```yaml
infer:
  img_path: 'data/img/1/test/'                                 
  json_path: 'data/annotations/1/captions_test.json'      
  model: 'models/one/1_freeze_dec/best/BEST_checkpoint_ms-coco.pth.tar'    
  vocab_path: 'data/vocab/1/vocab.pkl'                        
  prediction_path: 'infer/json/1_on_1_freeze_dec/'                          
  id2class_path: 'dataset/processed/id2class.json'             
```

then run:

```
python infer.py
```
* Step 6: Compute metrics on the new task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/1/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/1_on_1_freeze_dec/prediction.json'
```

### Distillation
* Step 2: Train model. Fine-tune from model 2to21, then from $HOME run :
```python3
python train.py --task_type one --task_name 1 --fine_tuning --distill
```
* Step 3: Infer captions to compute metrics by changing `infer` section in `config.yaml` file. Here we test this model on 2to21's test set:
```yaml
infer:
  img_path: 'data/img/2to21/test/'                              
  json_path: 'data/annotations/2to21/captions_test.json'       
  model: 'models/one/1_distill/best/BEST_checkpoint_ms-coco.pth.tar' 
  vocab_path: 'data/vocab/1/vocab.pkl'                     
  prediction_path: 'infer/json/1_on_2to21_distill/'                  
  id2class_path: 'dataset/processed/id2class.json'               
```

then run:

```
python infer.py
```
* Step 4: Compute metrics on the old task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/2to21/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/1_on_2to21_distill/prediction.json'
```
* Step 5: Infer captions to compute metrics 1's test set:
```yaml
infer:
  img_path: 'data/img/1/test/'                               
  json_path: 'data/annotations/1/captions_test.json'         
  model: 'models/one/1_distill/best/BEST_checkpoint_ms-coco.pth.tar'   
  vocab_path: 'data/vocab/1/vocab.pkl'                     
  prediction_path: 'infer/json/1_on_1_distill/'                          
  id2class_path: 'dataset/processed/id2class.json'           
```

then run:

```
python infer.py
```
* Step 6: Compute metrics on the new task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/1/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/1_on_1_distill/prediction.json'
```

## Addition of 5 classes at once
### Fine-tuning
* Step 1: Create data for this task. In `prepro/`, run
```bash
bash process_data.sh once once
```
* Step 2: Train model. Fine-tune from model 2to21, then from $HOME run :
```python3
python train.py --task_type once --task_name once --fine_tuning
```
* Step 3: Infer captions to compute metrics by changing `infer` section in `config.yaml` file. Here we test this model on 2to21's test set:
```yaml
infer:
  img_path: 'data/img/2to21/test/'                             
  json_path: 'data/annotations/2to21/captions_test.json'         
  model: 'models/once/once/best/BEST_checkpoint_ms-coco.pth.tar' 
  vocab_path: 'data/vocab/once/vocab.pkl'                     
  prediction_path: 'infer/json/once_on_2to21/'               
  id2class_path: 'dataset/processed/id2class.json'               
```

then run:

```
python infer.py
```
* Step 4: Compute metrics on the old task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/2to21/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/once_on_2to21/prediction.json'
```
* Step 5: Infer captions to compute metrics once's test set:
```yaml
infer:
  img_path: 'data/img/once/test/'                                  
  json_path: 'data/annotations/once/captions_test.json'            
  model: 'models/once/once/best/BEST_checkpoint_ms-coco.pth.tar'     
  vocab_path: 'data/vocab/once/vocab.pkl'                         
  prediction_path: 'infer/json/once_on_once/'                        
  id2class_path: 'dataset/processed/id2class.json'                 
```

then run:

```
python infer.py
```
* Step 6: Compute metrics on the new task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/once/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/once_on_once/prediction.json'
```

### Pseudo-labeling
* Step 2: Train model. Fine-tune from model 2to21, then from $HOME run :
```python3
python train.py --task_type once --task_name once --fine_tuning --lwf
```
* Step 3: Infer captions to compute metrics by changing `infer` section in `config.yaml` file. Here we test this model on 2to21's test set:
```yaml
infer:
  img_path: 'data/img/2to21/test/'                          
  json_path: 'data/annotations/2to21/captions_test.json'      
  model: 'models/once/once_lwf/best/BEST_checkpoint_ms-coco.pth.tar' 
  vocab_path: 'data/vocab/once/vocab.pkl'                     
  prediction_path: 'infer/json/once_on_2to21_lwf/'                 
  id2class_path: 'dataset/processed/id2class.json'                 
```

then run:

```
python infer.py
```
* Step 4: Compute metrics on the old task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/2to21/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/once_on_2to21_lwf/prediction.json'
```
* Step 5: Infer captions to compute metrics once's test set:
```yaml
infer:
  img_path: 'data/img/once/test/'                                 
  json_path: 'data/annotations/once/captions_test.json'         
  model: 'models/once/once_lwf/best/BEST_checkpoint_ms-coco.pth.tar'     
  vocab_path: 'data/vocab/once/vocab.pkl'                       
  prediction_path: 'infer/json/once_on_once_lwf/'                          
  id2class_path: 'dataset/processed/id2class.json'               
```

then run:

```
python infer.py
```
* Step 6: Compute metrics on the new task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/once/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/once_on_once_lwf/prediction.json'
```

### Freeze encoder
* Step 2: Train model. Fine-tune from model 2to21, then from $HOME run :
```python3
python train.py --task_type once --task_name once --fine_tuning --freeze_enc
```
* Step 3: Infer captions to compute metrics by changing `infer` section in `config.yaml` file. Here we test this model on 2to21's test set:
```yaml
infer:
  img_path: 'data/img/2to21/test/'                              
  json_path: 'data/annotations/2to21/captions_test.json'        
  model: 'models/once/once_freeze_enc/best/BEST_checkpoint_ms-coco.pth.tar' 
  vocab_path: 'data/vocab/once/vocab.pkl'                     
  prediction_path: 'infer/json/once_on_2to21_freeze_enc/'                 
  id2class_path: 'dataset/processed/id2class.json'                  
```

then run:

```
python infer.py
```
* Step 4: Compute metrics on the old task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/2to21/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/once_on_2to21_freeze_enc/prediction.json'
```
* Step 5: Infer captions to compute metrics once's test set:
```yaml
infer:
  img_path: 'data/img/once/test/'                                 
  json_path: 'data/annotations/once/captions_test.json'            
  model: 'models/once/once_freeze_enc/best/BEST_checkpoint_ms-coco.pth.tar'    
  vocab_path: 'data/vocab/once/vocab.pkl'                       
  prediction_path: 'infer/json/once_on_once_freeze_enc/'                          
  id2class_path: 'dataset/processed/id2class.json'                 
```

then run:

```
python infer.py
```
* Step 6: Compute metrics on the new task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/once/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/once_on_once_freeze_enc/prediction.json'
```

### Freeze decoder
* Step 2: Train model. Fine-tune from model 2to21, then from $HOME run :
```python3
python train.py --task_type once --task_name once --fine_tuning --freeze_dec
```
* Step 3: Infer captions to compute metrics by changing `infer` section in `config.yaml` file. Here we test this model on 2to21's test set:
```yaml
infer:
  img_path: 'data/img/2to21/test/'                            
  json_path: 'data/annotations/2to21/captions_test.json'        
  model: 'models/once/once_freeze_dec/best/BEST_checkpoint_ms-coco.pth.tar' 
  vocab_path: 'data/vocab/once/vocab.pkl'                     
  prediction_path: 'infer/json/once_on_2to21_freeze_dec/'                 
  id2class_path: 'dataset/processed/id2class.json'                  
```

then run:

```
python infer.py
```
* Step 4: Compute metrics on the old task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/2to21/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/once_on_2to21_freeze_dec/prediction.json'
```
* Step 5: Infer captions to compute metrics once's test set:
```yaml
infer:
  img_path: 'data/img/once/test/'                                 
  json_path: 'data/annotations/once/captions_test.json'          
  model: 'models/once/once_freeze_dec/best/BEST_checkpoint_ms-coco.pth.tar'   
  vocab_path: 'data/vocab/once/vocab.pkl'                         
  prediction_path: 'infer/json/once_on_once_freeze_dec/'                          
  id2class_path: 'dataset/processed/id2class.json'                 
```

then run:

```
python infer.py
```
* Step 6: Compute metrics on the new task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/once/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/once_on_once_freeze_dec/prediction.json'
```

### Distillation
* Step 2: Train model. Fine-tune from model 2to21, then from $HOME run :
```python3
python train.py --task_type once --task_name once --fine_tuning --distill
```
* Step 3: Infer captions to compute metrics by changing `infer` section in `config.yaml` file. Here we test this model on 2to21's test set:
```yaml
infer:
  img_path: 'data/img/2to21/test/'                          
  json_path: 'data/annotations/2to21/captions_test.json'        
  model: 'models/once/once_distill/best/BEST_checkpoint_ms-coco.pth.tar' 
  vocab_path: 'data/vocab/once/vocab.pkl'                     
  prediction_path: 'infer/json/once_on_2to21_distill/'                 
  id2class_path: 'dataset/processed/id2class.json'              
```

then run:

```
python infer.py
```
* Step 4: Compute metrics on the old task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/2to21/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/once_on_2to21_distill/prediction.json'
```
* Step 5: Infer captions to compute metrics once's test set:
```yaml
infer:
  img_path: 'data/img/once/test/'                                 
  json_path: 'data/annotations/once/captions_test.json'           
  model: 'models/once/once_distill/best/BEST_checkpoint_ms-coco.pth.tar'    
  vocab_path: 'data/vocab/once/vocab.pkl'                         
  prediction_path: 'infer/json/once_on_once_distill/'                          
  id2class_path: 'dataset/processed/id2class.json'                
```

then run:

```
python infer.py
```
* Step 6: Compute metrics on the new task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/once/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/once_on_once_distill/prediction.json'
```

## Addition of 5 classes sequentially
Because the last model that we obtain is from task 44 (bottle), we have to run testing on test splits of whole 5 new classes - Smultiple.
### Fine-tuning
* Step 1: Create data for this task. In `prepro/`, run
```bash
bash process_data.sh dump seq
```
* Step 2: Train model. Fine-tune from model 2to21, then from $HOME run :
```python3
python train.py --task_type seq --fine_tuning
```
* Step 3: Infer captions to compute metrics by changing `infer` section in `config.yaml` file. Here we test this model on 2to21's test set. Because the last model that we obtain is from task 44 (bottle), so we will compute metrics using task 44's model:
```yaml
infer:
  img_path: 'data/img/2to21/test/'                           
  json_path: 'data/annotations/2to21/captions_test.json'      
  model: 'models/seq/44_seq/best/BEST_checkpoint_ms-coco.pth.tar' 
  vocab_path: 'data/vocab/44/vocab.pkl'                    
  prediction_path: 'infer/json/44_seq_on_2to21/'                 
  id2class_path: 'dataset/processed/id2class.json'                
```

then run:

```
python infer.py
```
* Step 4: Compute metrics on the old task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/2to21/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/44_seq_on_2to21/prediction.json'
```
* Step 5: Infer captions to compute metrics once's test set:
```yaml
infer:
  img_path: 'data/img/once/test/'                                 
  json_path: 'data/annotations/once/captions_test.json'             
  model: 'models/seq/44_seq/best/BEST_checkpoint_ms-coco.pth.tar'   
  vocab_path: 'data/vocab/44/vocab.pkl'                          
  prediction_path: 'infer/json/44_seq_on_once/'                      
  id2class_path: 'dataset/processed/id2class.json'               
```

then run:

```
python infer.py
```
* Step 6: Compute metrics on the new task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/once/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/44_seq_on_once/prediction.json'
```

### Pseudo-labeling
* Step 2: Train model. Fine-tune from model 2to21, then from $HOME run :
```python3
python train.py --task_type seq --fine_tuning --lwf
```
* Step 3: Infer captions to compute metrics by changing `infer` section in `config.yaml` file. Here we test this model on 2to21's test set. Because the last model that we obtain is from task 44 (bottle), so we will compute metrics using task 44's model:
```yaml
infer:
  img_path: 'data/img/2to21/test/'                             
  json_path: 'data/annotations/2to21/captions_test.json'       
  model: 'models/seq/44_lwf_seq/best/BEST_checkpoint_ms-coco.pth.tar' 
  vocab_path: 'data/vocab/44/vocab.pkl'                      
  prediction_path: 'infer/json/44_seq_on_2to21_lwf/'                  
  id2class_path: 'dataset/processed/id2class.json'                 
```

then run:

```
python infer.py
```
* Step 4: Compute metrics on the old task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/2to21/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/44_seq_on_2to21_lwf/prediction.json'
```
* Step 5: Infer captions to compute metrics once's test set:
```yaml
infer:
  img_path: 'data/img/once/test/'                         
  json_path: 'data/annotations/once/captions_test.json'           
  model: 'models/seq/44_lwf_seq/best/BEST_checkpoint_ms-coco.pth.tar'   
  vocab_path: 'data/vocab/44/vocab.pkl'                       
  prediction_path: 'infer/json/44_seq_on_once_lwf/'                         
  id2class_path: 'dataset/processed/id2class.json'                
```

then run:

```
python infer.py
```
* Step 6: Compute metrics on the new task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/once/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/44_seq_on_once_lwf/prediction.json'
```

### Freeze encoder
* Step 2: Train model. Fine-tune from model 2to21, then from $HOME run :
```python3
python train.py --task_type seq --fine_tuning --freeze_enc
```
* Step 3: Infer captions to compute metrics by changing `infer` section in `config.yaml` file. Here we test this model on 2to21's test set. Because the last model that we obtain is from task 44 (bottle), so we will compute metrics using task 44's model:
```yaml
infer:
  img_path: 'data/img/2to21/test/'                            
  json_path: 'data/annotations/2to21/captions_test.json'       
  model: 'models/seq/44_freeze_enc_seq/best/BEST_checkpoint_ms-coco.pth.tar' 
  vocab_path: 'data/vocab/44/vocab.pkl'                      
  prediction_path: 'infer/json/44_seq_on_2to21_freeze_enc/'                  
  id2class_path: 'dataset/processed/id2class.json'                 
```

then run:

```
python infer.py
```
* Step 4: Compute metrics on the old task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/2to21/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/44_seq_on_2to21_freeze_enc/prediction.json'
```
* Step 5: Infer captions to compute metrics once's test set:
```yaml
infer:
  img_path: 'data/img/once/test/'                               
  json_path: 'data/annotations/once/captions_test.json'          
  model: 'models/seq/44_freeze_enc_seq/best/BEST_checkpoint_ms-coco.pth.tar'    
  vocab_path: 'data/vocab/44/vocab.pkl'                          
  prediction_path: 'infer/json/44_seq_on_once_freeze_enc/'                         
  id2class_path: 'dataset/processed/id2class.json'                  
```

then run:

```
python infer.py
```
* Step 6: Compute metrics on the new task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/once/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/44_seq_on_once_freeze_enc/prediction.json'
```

### Freeze decoder
* Step 2: Train model. Fine-tune from model 2to21, then from $HOME run :
```python3
python train.py --task_type seq --fine_tuning --freeze_dec
```
* Step 3: Infer captions to compute metrics by changing `infer` section in `config.yaml` file. Here we test this model on 2to21's test set. Because the last model that we obtain is from task 44 (bottle), so we will compute metrics using task 44's model:
```yaml
infer:
  img_path: 'data/img/2to21/test/'                              
  json_path: 'data/annotations/2to21/captions_test.json'         
  model: 'models/seq/44_freeze_dec_seq/best/BEST_checkpoint_ms-coco.pth.tar' 
  vocab_path: 'data/vocab/44/vocab.pkl'                       
  prediction_path: 'infer/json/44_seq_on_2to21_freeze_dec/'                  
  id2class_path: 'dataset/processed/id2class.json'                
```

then run:

```
python infer.py
```
* Step 4: Compute metrics on the old task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/2to21/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/44_seq_on_2to21_freeze_dec/prediction.json'
```
* Step 5: Infer captions to compute metrics once's test set:
```yaml
infer:
  img_path: 'data/img/once/test/'                              
  json_path: 'data/annotations/once/captions_test.json'            
  model: 'models/seq/44_freeze_dec_seq/best/BEST_checkpoint_ms-coco.pth.tar'    
  vocab_path: 'data/vocab/44/vocab.pkl'                         
  prediction_path: 'infer/json/44_seq_on_once_freeze_dec/'                         
  id2class_path: 'dataset/processed/id2class.json'                  
```

then run:

```
python infer.py
```
* Step 6: Compute metrics on the new task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/once/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/44_seq_on_once_freeze_dec/prediction.json'
```

### Distillation
* Step 2: Train model. Fine-tune from model 2to21, then from $HOME run :
```python3
python train.py --task_type seq --fine_tuning --distill
```
* Step 3: Infer captions to compute metrics by changing `infer` section in `config.yaml` file. Here we test this model on 2to21's test set. Because the last model that we obtain is from task 44 (bottle), so we will compute metrics using task 44's model:
```yaml
infer:
  img_path: 'data/img/2to21/test/'                               
  json_path: 'data/annotations/2to21/captions_test.json'      
  model: 'models/seq/44_distill_seq/best/BEST_checkpoint_ms-coco.pth.tar' 
  vocab_path: 'data/vocab/44/vocab.pkl'                      
  prediction_path: 'infer/json/44_seq_on_2to21_distill/'                  
  id2class_path: 'dataset/processed/id2class.json'               
```

then run:

```
python infer.py
```
* Step 4: Compute metrics on the old task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/2to21/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/44_seq_on_2to21_distill/prediction.json'
```
* Step 5: Infer captions to compute metrics once's test set:
```yaml
infer:
  img_path: 'data/img/once/test/'                                 
  json_path: 'data/annotations/once/captions_test.json'           
  model: 'models/seq/44_distill_seq/best/BEST_checkpoint_ms-coco.pth.tar'    
  vocab_path: 'data/vocab/44/vocab.pkl'                          
  prediction_path: 'infer/json/44_seq_on_once_distill/'                         
  id2class_path: 'dataset/processed/id2class.json'                  
```

then run:

```
python infer.py
```
* Step 6: Compute metrics on the new task by using `coco-caption/cocoEvalCapDemo.ipynb`. Modify:
```python3
annFile = 'your_path_to_$HOME/data/annotations/once/captions_test.json'
resFile = 'your_path_to_$HOME/infer/json/44_seq_on_once_distill/prediction.json'
```


