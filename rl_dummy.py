from sampling import sample_driver
from rl import HMMES, HMMPower
from trainer import Trainer
import matplotlib.pyplot as plt


n_sample = 10
n_state_a = 3
n_dim_a = 2
n_offspring = 15
n_episode = 100
duration = 1
std_init = 1.
std_decay = 0.9

regular_decay = lambda x: x*std_decay
identity_decay = lambda x: x

models = [(HMMPower, {'n_episode': n_episode}, 'power', regular_decay),
          (HMMES, {}, 'es', regular_decay), (HMMES, {'adapt_cov': True}, 'es_cov', identity_decay)]

demos_action = sample_driver(n_sample, n_state_a, n_dim_a)
trainers = []

for model, kwargs, name, decay_fn in models:
    m = model(demos_action, n_state_a, n_offspring, **kwargs)
    trainer = Trainer(m, demos_action, name)
    trainer.run(n_episode, n_offspring, std_init, decay_fn)
    trainers.append(trainer)

for t in trainers:
    plt.plot(t.rewards, label=t.name)
plt.xlabel('Rollout')
plt.ylabel('Cost')
plt.savefig('figures/costs.png', dpi=250, bbox_inches='tight')
plt.clf()
