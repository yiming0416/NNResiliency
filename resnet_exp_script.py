import os
import subprocess
from itertools import product, zip_longest, cycle

jobs = []
# baseline
jobs.append('python main.py --training_noise 0 --training_noise_mean 0 --device {device} --testing_noise 0 --testing_noise_mean 0 \n'
'python main.py --training_noise 0 --training_noise_mean 0 --device {device} --testing_noise 0 --testing_noise_mean 0 --test_quantization_levels 2 3 4 5 6 8 10 14 18 --testOnly --test_sample_num 1')

for std, mean in product([0], [-0.04, -0.02, -0.01, -0.004, 0.0, 0.004, 0.01, 0.02, 0.04]):
    jobs.append(f'python main.py --training_noise {std} --training_noise_mean {mean} --device {{device}} --testing_noise {std} --testing_noise_mean {mean} \n'
    f'python main.py --training_noise {std} --training_noise_mean {mean} --device {{device}} --testing_noise {std} --testing_noise_mean {mean} --test_quantization_levels 2 3 4 5 6 8 10 14 18 --testOnly --test_sample_num 1')

for std, mean in product([0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2], [0]):
    jobs.append(f'python main.py --training_noise {std} --training_noise_mean {mean} --device {{device}} --testing_noise {std} --testing_noise_mean {mean} \n'
    f'python main.py --training_noise {std} --training_noise_mean {mean} --device {{device}} --testing_noise {std} --testing_noise_mean {mean} --test_quantization_levels 2 3 4 5 6 8 10 14 18 --testOnly --test_sample_num 20')

for std, mean in zip([0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2], [0.0, 0.004, 0.01, 0.02, 0.04]):
    jobs.append(f'python main.py --training_noise {std} --training_noise_mean {mean} --device {{device}} --testing_noise {std} --testing_noise_mean {mean} \n'
    f'python main.py --training_noise {std} --training_noise_mean {mean} --device {{device}} --testing_noise {std} --testing_noise_mean {mean} --test_quantization_levels 2 3 4 5 6 8 10 14 18 --testOnly --test_sample_num 20')
# jobs.append('python main.py --training_noise 0 --training_noise_mean 0 --device {device} --testing_noise 0 --testing_noise_mean 0 --test_quantization_levels 2 3 4 5 6 8 10 14 18 --testOnly --test_sample_num 1')
# 
# for std, mean in product([0], [-0.04, -0.02, -0.01, -0.004, 0.0, 0.004, 0.01, 0.02, 0.04]):
#     jobs.append(f'python main.py --training_noise {std} --training_noise_mean {mean} --device {{device}} --testing_noise {std} --testing_noise_mean {mean} --test_quantization_levels 2 3 4 5 6 8 10 14 18 --testOnly --test_sample_num 1')
# 
# for std, mean in product([0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2], [0]):
#     jobs.append(f'python main.py --training_noise {std} --training_noise_mean {mean} --device {{device}} --testing_noise {std} --testing_noise_mean {mean} --test_quantization_levels 2 3 4 5 6 8 10 14 18 --testOnly --test_sample_num 20')
# 
# for std, mean in zip([0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2], [0.0, 0.004, 0.01, 0.02, 0.04]):
#     jobs.append(f'python main.py --training_noise {std} --training_noise_mean {mean} --device {{device}} --testing_noise {std} --testing_noise_mean {mean} --test_quantization_levels 2 3 4 5 6 8 10 14 18 --testOnly --test_sample_num 20')


concurrent_job_limit = 12

# https://stackoverflow.com/a/9809541
groups = [(subprocess.Popen(cmd.format(device=device), stdout=open(os.path.join("/tmp", str(i)), 'w'), stderr=subprocess.STDOUT, shell=True)
          for i, (cmd, device) in enumerate(zip(jobs, cycle(range(6)))))] * concurrent_job_limit

for processes in zip_longest(*groups):
    for p in filter(None, processes):
        print(p.args)
        p.wait()
