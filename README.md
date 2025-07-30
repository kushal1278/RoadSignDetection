# RoadSignDetection
A simple multi-layer CNN used for recognizing road signs and assing a label to it.

## How to use
### If training:
1) Run either ```train_with_augm.py``` or ```train_without_augm.py``` as such
   1) ```python train_with_augm.py``` for performing data augmentation and training
   2) ```python train_without_augm.py``` for training right away
      
2) Then run ```demo.py``` in the following way
   ```python demo.py <mode>```
   
   where ```<mode>``` is
   1) ```augm``` for using the model trained on AUGMENTED DATA
   2) ```naugm``` for using the model trained on UNAUGMENTED DATA
   3) if nothing is provided, it will use model trained on AUGMENTED DATA

### If not training:
1) run ```demo.py``` as described above
