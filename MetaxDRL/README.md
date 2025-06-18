# Metax Deep Reinforcement Learning Framework

English | [中文版](README_zh.md)

MetaxDRL is a lightweight framework designed specifically for Deep Reinforcement Learning (DRL), offering high-quality, single-file implementations of various mainstream reinforcement learning algorithms. Its core design philosophy centers on simplicity, readability, and practicality, enabling users to quickly grasp algorithm principles while supporting flexible customization and experimental innovation.

## Key Features
- **Online Algorithm**: Each online algorithm module is meticulously crafted with a clear and concise code structure, avoiding redundant encapsulation to preserve the intuitive expression of algorithmic logic. This "de-engineered" approach is particularly suitable for teaching, rapid prototyping, and research exploration, allowing researchers and developers to focus on algorithm optimization and innovation.

- **Offline Algorithm**: Leveraging the d3rlpy framework, MetaxDRL provides efficient support for offline algorithms, harnessing the robust performance of Metax C-series GPUs to demonstrate compatibility with mainstream frameworks.

- **Environment Compatibility and Performance**: MetaxDRL supports a wide range of classic and modern reinforcement learning environments, delivering excellent runtime performance and user-friendly training and evaluation functionalities, catering to both beginners and seasoned researchers.

## Application Scenarios
With its simple yet powerful design, MetaxDRL is ideal for the following scenarios:

- **Teaching and Learning**: Clear single-file implementations help students and beginners quickly understand the core principles of reinforcement learning algorithms.

- **Research Exploration**: Flexible code structure facilitates algorithm improvements and experimental innovation for researchers.


- **Rapid Prototyping**: Enables fast development and testing, empowering developers to validate new ideas efficiently.

## Algorithms Implemented
<table align="center">
    <tr>
        <th rowspan="2" align="center">Algorithm</th>
        <th colspan="2" align="center">type</th>
        <th rowspan="2" align="center">Implemented</th>
        <th colspan="2" align="center">Action Space</th>
        <th colspan="3" align="center">Device</th>
        <th rowspan="2" align="center">Example Environment</th>
        <th rowspan="2" align="center">Reward</th>
        <th colspan="3" align="center">Speed(FPS)</th>
    </tr>
    <tr>
        <th align="center">Online</th>
        <th align="center">Offline</th>
        <th align="center">Discrete</th>
        <th align="center">Continuous</th>
        <th align="center">CPU</th>
        <th align="center">C500</th>
        <th align="center">GPU</th>
        <th align="center">CPU</th>
        <th align="center">C500</th>
        <th align="center">GPU</th>
    </tr>
    <tr>
        <td align="left"><a href="https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf">✅A2C</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/a2c/train.py">train.py</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
        <td align="center">195</td>
        <td align="center">2527</td>
        <td align="center">1052</td>
        <td align="center">430</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/1602.01783">✅A3C</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/a3c/train.py">train.py</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
        <td align="center">195</td>
        <td align="center">114028</td>
        <td align="center">375080</td>
        <td align="center">123180</td>
    </tr>
    <tr>
        <td align="left"><a href="https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf">✅AC</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/ac/train.py">train.py</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
        <td align="center">195</td>
        <td align="center">1590</td>
        <td align="center">843</td>
        <td align="center">330</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/1707.06887">✅C51</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/c51/train.py">train.py</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
        <td align="center">195</td>
        <td align="center">32598</td>
        <td align="center">15796</td>
        <td align="center">19836</td>
    </tr>
    <tr>
        <td align="left"><a href="https://proceedings.mlr.press/v32/silver14.pdf">✅DDPG</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/ddpg/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/half_cheetah/">HalfCheetah-v4</a></td>
        <td align="center">4800</td>
        <td align="center">33746</td>
        <td align="center">98095</td>
        <td align="center">41333</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/1509.06461">✅Double DQN</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/double_dqn/train.py">train.py</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
        <td align="center">195</td>
        <td align="center">151048</td>
        <td align="center">206638</td>
        <td align="center">121803</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/1312.5602">✅DQN</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/dqn/train.py">train.py</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
        <td align="center">195</td>
        <td align="center">164655</td>
        <td align="center">222177</td>
        <td align="center">127998</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/1509.06461">✅Dueling DQN</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/dueling_dqn/train.py">train.py</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
        <td align="center">195</td>
        <td align="center">134419</td>
        <td align="center">192030</td>
        <td align="center">91482</td>
    </tr>
    <tr>
        <td align="left"><a href="https://link.springer.com/content/pdf/10.1007/BF00992696.pdf">✅Policy Gradient</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/policy_gradient/train.py">train.py</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
        <td align="center">195</td>
        <td align="center">17</td>
        <td align="center">10</td>
        <td align="center">5</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/1707.06347">✅PPO</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/ppo/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/half_cheetah/">HalfCheetah-v4</a></td>
        <td align="center">4800</td>
        <td align="center">15315</td>
        <td align="center">7064</td>
        <td align="center">2780</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/1812.05905">✅SAC</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/sac/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/half_cheetah/">HalfCheetah-v4</a></td>
        <td align="center">4800</td>
        <td align="center">737</td>
        <td align="center">1108</td>
        <td align="center">610</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/1802.09477">✅TD3</a></td>
        <td align="center">✅</td>
        <td align="center">/</td>
        <td align="center"><a href="./example/online/td3/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/half_cheetah/">HalfCheetah-v4</a></td>
        <td align="center">4800</td>
        <td align="center">52359</td>
        <td align="center">77272</td>
        <td align="center">45469</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/2006.09359">✅AWAC</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center"><a href="./example/offline/awac/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/hopper/">Hopper-v0</a></td>
        <td align="center">3000</td>
        <td align="center">19</td>
        <td align="center">66</td>
        <td align="center">30</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/1812.02900">✅BCQ</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center"><a href="./example/offline/bcq/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/hopper/">Hopper-v0</a></td>
        <td align="center">3000</td>
        <td align="center">9</td>
        <td align="center">95</td>
        <td align="center">46</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/1906.00949">✅BEAR</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center"><a href="./example/offline/bear/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/hopper/">Hopper-v0</a></td>
        <td align="center">3000</td>
        <td align="center">32</td>
        <td align="center">108</td>
        <td align="center">43</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/2006.04779">✅CQL</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center"><a href="./example/offline/cql/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/hopper/">Hopper-v0</a></td>
        <td align="center">3000</td>
        <td align="center">16</td>
        <td align="center">77</td>
        <td align="center">34</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/2006.15134">✅CRR</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center"><a href="./example/offline/crr/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/hopper/">Hopper-v0</a></td>
        <td align="center">3000</td>
        <td align="center">70</td>
        <td align="center">159</td>
        <td align="center">62</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/2106.01345">✅Decision Transformer</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center"><a href="./example/offline/decision_transformer/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/hopper/">Hopper-v0</a></td>
        <td align="center">3000</td>
        <td align="center">14</td>
        <td align="center">126</td>
        <td align="center">54</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/2110.06169">✅IQL</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center"><a href="./example/offline/iql/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/hopper/">Hopper-v0</a></td>
        <td align="center">3000</td>
        <td align="center">109</td>
        <td align="center">150</td>
        <td align="center">61</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/2011.07213">✅PLAS</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center"><a href="./example/offline/plas/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/hopper/">Hopper-v0</a></td>
        <td align="center">3000</td>
        <td align="center">127</td>
        <td align="center">323</td>
        <td align="center">113</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/1802.09477">✅TD3</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center"><a href="./example/offline/td3/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/hopper/">Hopper-v0</a></td>
        <td align="center">3000</td>
        <td align="center">137</td>
        <td align="center">194</td>
        <td align="center">80</td>
    </tr>
    <tr>
        <td align="left"><a href="https://arxiv.org/pdf/2106.06860">✅TD3+BC</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center"><a href="./example/offline/td3_plus_bc/train.py">train.py</a></td>
        <td align="center">/</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center">✅</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/hopper/">Hopper-v0</a></td>
        <td align="center">3000</td>
        <td align="center">131</td>
        <td align="center">188</td>
        <td align="center">80</td>
    </tr>
</table>
⚠️ NOTE:The FPS test data for the above offline algorithm is sourced from the print output of the d3rlpy framework, which differs from the testing method of the online algorithm and is provided for reference only.

## Installation

Please refer to [install.md](./docs/install.md) for environment setup.

## Usage

Please refer to [usage.md](./docs/usage.md) for usage.

## Acknowledgements

The offline algorithms in this project rely on [d3rlpy](https://github.com/takuseno/d3rlpy/tree/master)
