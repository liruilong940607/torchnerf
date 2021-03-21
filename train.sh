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
CONFIG=$1
SCENE=$2
DATA_ROOT=$3
ROOT_DIR=./torchnerf_ckpt/"$CONFIG"

if [[ "lego chair drums ficus hotdog materials mic ship" =~ "$SCENE" ]]; then
    DATA_FOLDER="nerf_synthetic"
elif [[ "pinecone vasedeck" =~ "$SCENE" ]]; then
    DATA_FOLDER="nerf_real_360"
else
    DATA_FOLDER="nerf_llff_data"
fi

# launch training job.
python -m torchnerf.train \
    --data_dir="$DATA_ROOT"/"$DATA_FOLDER"/"$SCENE" \
    --train_dir="$ROOT_DIR"/"$SCENE" \
    --config=configs/"$CONFIG"
