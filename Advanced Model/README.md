# Political bias detection using LSTM

Author: Yue Pei

This directory is focused on this classification task using LSTM

Please note that I didn't put the datasets in this directory. They are too large.

In this work, two LSTM architectures are proposed and tested over 11 datasets

This directory contains several Python notebooks with the developed code 

### Final Result 

| predicted          | 2020_data | 2021_data | Police | Police_5yr | Covid  | Gun    | Gun_5yr | Vaccine | Vaccine_5yr | Election_Fraud | Election_Fraud_5yr |
| ------------------ | --------- | --------- | ------ | ---------- | ------ | ------ | ------- | ------- | ----------- | -------------- | ------------------ |
| 2020_data          | 88.24%    | N/A       | 83%    | 87.36%     | 91.66% | 80.73% | 86.30%  | 92%     | 92%         | 90%            | 90.67%             |
| 2021_data          | N/A       | 88.48%    | 70.56% | 76.33%     | 94%    | 59.01% | 65%     | 92%     | 93%         | 89%            | 89.87%             |
| Police             | N/A       | 73.97%    | N/A    | N/A        | 76.38% | 77.00% | 79%     | 79%     | 80%         | 82%            | 81.58%             |
| Police_5yr         | N/A       | 68.87%    | N/A    | N/A        | 83.33% | 73.52% | 76.67%  | 76%     | 76%         | 80%            | 80%                |
| Covid              | N/A       | 35.20%    | 45.74% | 43.36%     | N/A    | 55.38% | 54.12%  | 43%     | 43%         | 58%            | 58%                |
| Gun                | N/A       | 59.32%    | 66.68% | 66.71%     | 68.05% | N/A    | N/A     | 67%     | 67%         | 69%            | 69%                |
| Gun_5yr            | N/A       | 65.47%    | 69.29% | 70.65%     | 70.83% | N/A    | N/A     | 73%     | 73%         | 71%            | 71%                |
| Vaccine            | N/A       | 69.42%    | 58.78% | 63.06%     | 86.11% | 56.94% | 62.07%  | N/A     | N/A         | 75%            | 75%                |
| Vaccine_5yr        | N/A       | 68.42%    | 59.72% | 63.48%     | 73.61% | 60%    | 65.31%  | N/A     | N/A         | 75%            | 75.48%             |
| Election_Fraud     | N/A       | 38.00%    | 43.97% | 42.69%     | 47.22% | 50.86% | 50.39%  | 45%     | 45%         | N/A            | N/A                |
| Election_Fraud_5yr | N/A       | 34.87%    | 42.22% | 41.14%     | 37.50% | 49.36% | 48.79%  | 42%     | 41.88%      | N/A            | N/A                |

### Notebook explanation

* `LSTM_training_v1.ipynb:` Training LSTM for 11 datasets.

* `LSTM_training_v2.ipynb:` Trying different LSTM architecture.

* `data_analysis:` Cleaning and preprocessing the datasets. Also did some analysis, e.g., N-grams and Word Cloud/

  

