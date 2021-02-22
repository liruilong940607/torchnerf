# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -e
set -x

conda create -n torchnerf python=3.7
conda activate torchnerf

conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch-nightly
pip install -r torchnerf/requirements.txt

python -m torchnerf.train \
  --data_dir=torchnerf/example_data \
  --train_dir=/tmp/torchnerf_test \
  --max_steps=5 \
  --factor=2 \
  --batch_size=512