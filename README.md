# SegNet- tensorflow implementation
Training loss, training accuracy, validation loss, validation accuracy, test accuracy and MoI for different models.


|     Statistics     | SegNet Scartch| SegNet Vgg | SegNet Bayes Scratch | SegNet Bayes Vgg |
| ------------------ | ------------- | ---------- | -------------------- | ---------------- |
| Training Loss      |    0.0205     |  0.0320    |   0.0545             |   0.068          |
| Training Accuracy  |    99.19%     |  98.72%    |   97.89%             |   97.4%          |
| Validation Loss    |    0.9031     |  0.8974    |   0.5352             |   0.487          |
| Validation Accuracy|    87.12%     |  87.69%    |   90.33%             |   90.8%          |
| Test Accuracy      |    81.33%     |  81.79%    |   82.69%(82.67%)     |   84.25%(84.04%)(84.23%)| 
| Test MoI           |    42.50%     |  43.77%    |   47.50%(47.50%)     |   47.64%(47.89%)(47.89%)|

The second bracket is without applying dropout in test time. The third bracket is applying dropout but with using Max Vote instead of mean. 


Class average accuracy for test image for different model:

| Method       | Sky   | Building| Column Pole| Road | Side-Walk | Tree | Sign Symbol| Fence | Car  | Pedestrain | Bicyclist |
| ------------ | ------| ------- | -----------| ---- | --------- | ---- | -----------| ----- | ---  | ---------- | --------- |
| +scratch     | 89.46 | 68.59   | 9.30       |85.72 | 55.25     |59.78 | 18.42      | 11.53 |62.08 | 15.67      | 9.65      |
| +vgg         | 89.42 | 68.66   | 11.02      |87.28 | 62.66     |59.10 | 16.53      | 12.51 |57.96 | 18.82      | 17.58     |
| +scratch(bayes) ALL| 90.05 | 69.13   | 15.75      |88.66 | 65.22     |59.79 | 25.85      | 14.68 |68.95 | 24.73      | 22.44     |
| +scratch(bayes) 0.5| 89.82 | 69.51   | 12.51      |88.78 | 65.44     |60.30 | 24.81      | 15.74 |70.46 | 24.41      | 18.09     |
| +vgg(bayes) 0.5| 90.18 | 70.03   | 10.64      |90.50 | 71.26     |62.22 | 30.25      | 13.90 |59.82 | 27.09      | 17.63     |
| +vgg(bayes) ALL| 89.97 | 70.19   | 11.43      |89.90 | 70.09     |61.19 | 31.50      | 11.70 |61.02 | 28.70      | 21.03     |
| +vgg(bayes) 0.5 MAXVOTE| 90.25 | 70.02   | 11.50      |90.39 | 71.20     |62.07 | 30.44      | 13.82 |59.84 | 27.89      | 19.08     |







Results are shown as percentage(%)



