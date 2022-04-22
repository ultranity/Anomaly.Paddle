# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse

import paddle

from model import get_model
import datasets.mvtec as mvtec

def parse_args():
    parser = argparse.ArgumentParser(description='Model export.')
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the exported model',
        type=str,
        default=None)
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for export',
        type=str,
        default=None)
    parser.add_argument("--category", type=str, default='leather', help="category name for MvTec AD dataset")
    parser.add_argument("--arch", type=str, default='resnet18', help="backbone model arch, one of [resnet18, resnet50, wide_resnet50_2]")
    parser.add_argument("--k", type=int, default=100, help="feature used")
    parser.add_argument("--method", type=str, default='sample', help="projection method, one of [sample,ortho]")
    parser.add_argument('--img_size', type=int, default=256)

    return parser.parse_args()


def main():
    args = parse_args()
    print(args)
    if args.save_dir == None:
        args.save_dir = f"output/{args.method}_{args.arch}_{args.k}"
    # build model
    model = get_model(args.method)(arch=args.arch, pretrained=False, k=args.k, method= args.method)
    class_name = args.category
    assert class_name in mvtec.CLASS_NAMES
    model_path = args.model_path or f'{args.save_dir}/{class_name}.pdparams'
    state = paddle.load(model_path)
    model.model.set_dict(state.pop("params"))
    model.load(state["stats"])
    model.eval()
    paddle.save(state["stats"], os.path.join(args.save_dir, 'stats'))

    shape = [-1, 3, args.img_size, args.img_size]
    model = paddle.jit.to_static(
        model,
        input_spec=[paddle.static.InputSpec(shape=shape, dtype='float32')])
    save_path = os.path.join(args.save_dir, 'model')
    paddle.jit.save(model, save_path)
    print(f'Model is saved in {args.save_dir}.')


if __name__ == '__main__':
    main()